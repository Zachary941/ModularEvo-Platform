"""
步骤3：函数式动态合并逻辑

核心功能：
1. compute_merged_params: 用 Router 输出的 α 和预存的 Task Vectors 计算合并后的模型参数
   （保留计算图，可微分——梯度从 loss 经合并参数回传到 Router 的 α）
2. merged_forward: 完整的前向推理管线（functional_call + 后处理 + 分类头）
3. compute_merge_loss: 按任务拆分 Batch 并计算合并模型的分类 loss

关键技术：
- 使用 torch.func.functional_call（PyTorch 2.0+）实现无状态前向传播
- α 必须保持 requires_grad=True，确保梯度链完整
- 隐状态后处理（Mean Pooling + LayerNorm）必须与微调模型 GPTNeoWithClassificationHead.forward 一致
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .config import TASK_NAMES, HIDDEN_SIZE, TASK_NAME_TO_IDX

# 优先使用 torch.func.functional_call (PyTorch 2.0+)，回退到旧版
try:
    from torch.func import functional_call
except ImportError:
    from torch.nn.utils.stateless import functional_call


def compute_merged_params(
    base_params: Dict[str, torch.Tensor],
    task_vectors: Dict[str, Dict[str, torch.Tensor]],
    alphas: torch.Tensor,
    task_names: List[str] = TASK_NAMES,
) -> Dict[str, torch.Tensor]:
    """
    计算合并后的模型参数（保留计算图，可微分）。
    
    merged_param[name] = base_param[name] + Σ_i (α_i * τ_i[name])
    
    Args:
        base_params: Dict[str, Tensor] — GPTNeoForCausalLM 的参数（不含 base_model. 前缀）
        task_vectors: Dict[str, Dict[str, Tensor]] — 各任务向量
        alphas: Tensor [num_tasks] — Router 输出的合并权重（带梯度！）
        task_names: List[str] — 任务名列表，与 alphas 索引对应
    
    Returns:
        merged_params: Dict[str, Tensor] — 合并后的参数（保留梯度链到 alphas）
    """
    merged_params = {}
    for name, base_param in base_params.items():
        # 累加所有任务的加权 task vector
        delta = torch.zeros_like(base_param)
        for i, task_name in enumerate(task_names):
            if name in task_vectors[task_name]:
                tau_param = task_vectors[task_name][name]
                # alphas[i] * tau 保持计算图
                delta = delta + alphas[i] * tau_param
        merged_params[name] = base_param + delta
    return merged_params


def get_hidden_states(
    base_model: nn.Module,
    merged_params: Dict[str, torch.Tensor],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """
    使用 functional_call 进行无状态前向传播，获取最后一层隐状态。
    
    Args:
        base_model: GPTNeoForCausalLM 实例（仅提供计算图结构，不使用其权重）
        merged_params: 合并后的参数字典
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
    
    Returns:
        last_hidden_state: [Batch, Seq_Len, Hidden_Size]
    """
    output = functional_call(
        base_model,
        merged_params,
        args=(),
        kwargs={
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
    )
    return output.hidden_states[-1]


def postprocess_hidden_states(
    last_hidden_state: torch.Tensor,
    hidden_size: int = HIDDEN_SIZE,
) -> torch.Tensor:
    """
    对最后一层隐状态做与微调模型一致的后处理：Mean Pooling + LayerNorm。
    
    与 GPTNeoWithClassificationHead.forward 中的逻辑完全一致：
        sentence_representation = torch.mean(last_hidden_state, dim=1)
        layer_norm = nn.LayerNorm(hidden_size).to(device)
        sentence_representation = layer_norm(sentence_representation)
    
    注意：微调代码中 LayerNorm 在 forward 中每次新建（使用默认参数），
    因此这里也使用未训练的 LayerNorm（默认 weight=1, bias=0）。
    
    Args:
        last_hidden_state: [Batch, Seq, Hidden]
    
    Returns:
        sentence_representation: [Batch, Hidden]
    """
    # Mean Pooling over sequence dimension
    sentence_repr = torch.mean(last_hidden_state, dim=1)   # [Batch, Hidden]
    
    # LayerNorm（默认参数，与微调代码一致）
    ln = nn.LayerNorm(hidden_size).to(last_hidden_state.device)
    sentence_repr = ln(sentence_repr)                       # [Batch, Hidden]
    
    return sentence_repr


def merged_forward(
    base_model: nn.Module,
    merged_params: Dict[str, torch.Tensor],
    heads: Dict[str, nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    task_ids: List[str],
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    完整的合并模型前向推理：functional_call → 后处理 → 按任务拆分 → 分类头。
    
    Args:
        base_model: GPTNeoForCausalLM（提供计算图结构）
        merged_params: 合并后的参数
        heads: Dict[str, nn.Sequential] — 各任务的冻结分类头
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
        task_ids: List[str] 长度=Batch，每条样本的任务名
    
    Returns:
        all_logits: Dict[task_name, Tensor] — 各任务子 batch 的 logits
        all_indices: Dict[task_name, Tensor] — 各任务子 batch 在原 Batch 中的索引
    """
    # 1. 获取隐状态
    last_hidden = get_hidden_states(base_model, merged_params, input_ids, attention_mask)
    
    # 2. 后处理
    sentence_repr = postprocess_hidden_states(last_hidden)   # [Batch, Hidden]
    
    # 3. 按任务拆分并过分类头
    all_logits = {}
    all_indices = {}
    
    for task_name in set(task_ids):
        # 找到属于该任务的样本索引
        indices = [i for i, t in enumerate(task_ids) if t == task_name]
        if not indices:
            continue
        
        idx_tensor = torch.tensor(indices, device=sentence_repr.device)
        sub_repr = sentence_repr[idx_tensor]                  # [sub_batch, Hidden]
        
        # 过对应的分类头
        logits = heads[task_name](sub_repr)                   # [sub_batch, num_classes]
        
        all_logits[task_name] = logits
        all_indices[task_name] = idx_tensor
    
    return all_logits, all_indices


def compute_merge_loss(
    base_model: nn.Module,
    merged_params: Dict[str, torch.Tensor],
    heads: Dict[str, nn.Module],
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    task_ids: List[str],
    task_loss_reduction: str = 'sample',
) -> torch.Tensor:
    """
    计算合并模型的分类 loss。
    
    按 task_ids 将 Batch 拆分为各任务的 sub-batch，
    使用对应的冻结分类头计算 CE Loss，然后对各子任务的 loss 取平均。
    
    Args:
        base_model: GPTNeoForCausalLM
        merged_params: 合并后的参数（保留梯度链到 alphas）
        heads: Dict[str, nn.Sequential] — 各任务的冻结分类头
        input_ids: [Batch, Seq_Len]
        attention_mask: [Batch, Seq_Len]
        labels: [Batch] — 分类标签
        task_ids: List[str] 长度=Batch，每条样本的任务名
        task_loss_reduction:
            - 'sample': 按全部样本平均，多数任务贡献更大梯度
            - 'task_balanced': 每个出现的任务先各自平均，再对任务做平均
    
    Returns:
        merge_loss: 标量 Tensor（可微分，保留梯度链到 alphas）
    """
    # 前向推理得到各任务的 logits
    all_logits, all_indices = merged_forward(
        base_model, merged_params, heads,
        input_ids, attention_mask, task_ids
    )
    
    if not all_logits:
        return torch.tensor(0.0, device=input_ids.device, requires_grad=True)

    if task_loss_reduction == 'sample':
        loss_fn = nn.CrossEntropyLoss(reduction='sum')
        total_loss = torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        total_samples = 0

        for task_name, logits in all_logits.items():
            idx = all_indices[task_name]
            sub_labels = labels[idx]
            task_loss = loss_fn(logits, sub_labels)
            total_loss = total_loss + task_loss
            total_samples += len(sub_labels)

        if total_samples == 0:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        return total_loss / total_samples

    if task_loss_reduction == 'task_balanced':
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        task_losses = []

        for task_name, logits in all_logits.items():
            idx = all_indices[task_name]
            sub_labels = labels[idx]
            if len(sub_labels) == 0:
                continue
            task_losses.append(loss_fn(logits, sub_labels))

        if not task_losses:
            return torch.tensor(0.0, device=input_ids.device, requires_grad=True)
        return torch.stack(task_losses).mean()

    raise ValueError(f"未知的 task_loss_reduction: {task_loss_reduction}")


def compute_total_loss(
    merge_loss: torch.Tensor,
    task_cls_loss: torch.Tensor,
    lambda_task_cls: float = 0.5,
) -> torch.Tensor:
    """
    计算总 loss = merge_loss + λ * task_cls_loss
    
    Args:
        merge_loss: 合并模型的分类 loss
        task_cls_loss: Router Task Classifier 的 CE loss
        lambda_task_cls: 辅助损失的权重系数
    
    Returns:
        total_loss: 总 loss
    """
    return merge_loss + lambda_task_cls * task_cls_loss
