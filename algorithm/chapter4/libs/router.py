"""
步骤2：Router 网络定义（双头架构 + 兼容旧 checkpoint）

当前默认训练策略使用双支路 merge head：
- dataset_embed 单独编码
- task_dist 单独编码
- 两支投到同宽空间后再融合生成 α

同时保留旧版 concat 结构，保证历史 checkpoint 仍能加载与评测。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .config import HIDDEN_SIZE, NUM_TASKS


ROUTER_VARIANT_HYBRID_DUAL = 'hybrid_dual_branch'
ROUTER_VARIANT_HYBRID = 'hybrid_embed_taskdist'
ROUTER_VARIANT_TASKDIST_ONLY = 'taskdist_only_small'
ROUTER_VARIANT_EMBED_ONLY = 'embed_only_legacy'
DEFAULT_ROUTER_VARIANT = ROUTER_VARIANT_HYBRID_DUAL


def infer_router_variant_from_state_dict(
    state_dict,
    hidden_size: int = HIDDEN_SIZE,
    num_tasks: int = NUM_TASKS,
) -> str:
    """根据 checkpoint 中 merge_head 的形状推断 Router 结构。"""
    embed_branch_weight = state_dict.get('embed_branch.0.weight')
    task_branch_weight = state_dict.get('task_branch.0.weight')
    fusion_head_weight = state_dict.get('fusion_head.0.weight')
    if (
        embed_branch_weight is not None
        and task_branch_weight is not None
        and fusion_head_weight is not None
    ):
        if (
            embed_branch_weight.shape[1] == hidden_size
            and task_branch_weight.shape[1] == num_tasks
            and fusion_head_weight.shape[1] == 256
        ):
            return ROUTER_VARIANT_HYBRID_DUAL

    first_weight = state_dict.get('merge_head.0.weight')
    if first_weight is None:
        return DEFAULT_ROUTER_VARIANT

    out_dim, in_dim = first_weight.shape
    if in_dim == hidden_size + num_tasks and out_dim == 256:
        return ROUTER_VARIANT_HYBRID
    if in_dim == num_tasks and out_dim == 64:
        return ROUTER_VARIANT_TASKDIST_ONLY
    if in_dim == hidden_size and out_dim == 128:
        return ROUTER_VARIANT_EMBED_ONLY

    raise ValueError(
        f'无法从 merge_head.0.weight 形状 {tuple(first_weight.shape)} 推断 Router 结构'
    )


class Router(nn.Module):
    """
    动态路由网络：分析输入数据的分布，输出模型合并权重（数据集级别）和样本-任务归属。

    支持四种兼容结构：
    1. hybrid_dual_branch: 默认的新训练结构，dataset_embed / task_dist 双支路编码后融合
    2. hybrid_embed_taskdist: 旧版 concat 结构，直接拼接 dataset_embed 和 task_dist
    3. taskdist_only_small: 当前 109,864 参数的小结构，只接收 task_dist
    4. embed_only_legacy: 更早期的复原结构，只接收 dataset_embed
    
    Args:
        embedding_layer: GPTNeoForCausalLM 的 Embedding 层（冻结）
                         即 base_model.transformer.wte
        hidden_size: Embedding 维度（768 for GPT-Neo 125M）
        num_tasks: 任务数量（4：code, langid, law, math）
        dropout: Dropout 比率
    """
    
    def __init__(
        self,
        embedding_layer: nn.Embedding,
        hidden_size: int = HIDDEN_SIZE,
        num_tasks: int = NUM_TASKS,
        dropout: float = 0.1,
        variant: str = DEFAULT_ROUTER_VARIANT,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_tasks = num_tasks
        self.variant = variant
        
        # 共享 Encoder：使用冻结的 GPT-Neo Embedding
        self.embedding = embedding_layer
        # 冻结 Embedding 层
        for param in self.embedding.parameters():
            param.requires_grad = False
        
        self.use_dataset_embed = variant in {
            ROUTER_VARIANT_HYBRID_DUAL,
            ROUTER_VARIANT_HYBRID,
            ROUTER_VARIANT_EMBED_ONLY,
        }
        self.use_task_dist = variant in {
            ROUTER_VARIANT_HYBRID_DUAL,
            ROUTER_VARIANT_HYBRID,
            ROUTER_VARIANT_TASKDIST_ONLY,
        }
        self._trainable_modules = []

        if variant == ROUTER_VARIANT_HYBRID_DUAL:
            branch_width = 128
            self.embed_branch = nn.Sequential(
                nn.Linear(hidden_size, 192),
                nn.LayerNorm(192),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(192, branch_width),
                nn.LayerNorm(branch_width),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.task_branch = nn.Sequential(
                nn.Linear(num_tasks, branch_width),
                nn.LayerNorm(branch_width),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(branch_width, branch_width),
                nn.LayerNorm(branch_width),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            self.fusion_head = nn.Sequential(
                nn.Linear(branch_width * 2, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
                nn.Sigmoid(),
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
            )
            self._trainable_modules.extend([
                self.embed_branch,
                self.task_branch,
                self.fusion_head,
                self.task_classifier,
            ])
        elif variant == ROUTER_VARIANT_HYBRID:
            merge_input_dim = hidden_size + num_tasks
            self.merge_head = nn.Sequential(
                nn.Linear(merge_input_dim, 256),
                nn.LayerNorm(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
                nn.Sigmoid(),
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
            )
            self._trainable_modules.extend([
                self.merge_head,
                self.task_classifier,
            ])
        elif variant == ROUTER_VARIANT_TASKDIST_ONLY:
            self.merge_head = nn.Sequential(
                nn.Linear(num_tasks, 64),
                nn.LayerNorm(64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, num_tasks),
                nn.Sigmoid(),
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
            )
            self._trainable_modules.extend([
                self.merge_head,
                self.task_classifier,
            ])
        elif variant == ROUTER_VARIANT_EMBED_ONLY:
            self.merge_head = nn.Sequential(
                nn.Linear(hidden_size, 128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
                nn.Sigmoid(),
            )
            self.task_classifier = nn.Sequential(
                nn.Linear(hidden_size, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, num_tasks),
            )
            self._trainable_modules.extend([
                self.merge_head,
                self.task_classifier,
            ])
        else:
            raise ValueError(f'未知的 Router 结构: {variant}')
        
        # 初始化可训练参数
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 初始化可训练层的权重"""
        for module in self._trainable_modules:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    @staticmethod
    def _ensure_2d(tensor: torch.Tensor) -> torch.Tensor:
        if tensor is None:
            return None
        if tensor.dim() == 1:
            return tensor.unsqueeze(0)
        return tensor

    def _prepare_aggregated_inputs(
        self,
        dataset_embed: torch.Tensor,
        dataset_task_dist: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        dataset_embed = self._ensure_2d(dataset_embed)
        dataset_task_dist = self._ensure_2d(dataset_task_dist)

        if self.use_dataset_embed and dataset_embed is None:
            raise ValueError('当前 Router 结构要求提供 dataset_embed')
        if self.use_task_dist and dataset_task_dist is None:
            raise ValueError('当前 Router 结构要求提供 dataset_task_dist')

        batch_sizes = []
        if dataset_embed is not None:
            batch_sizes.append(dataset_embed.size(0))
        if dataset_task_dist is not None:
            batch_sizes.append(dataset_task_dist.size(0))
        if not batch_sizes:
            raise ValueError('dataset_embed 和 dataset_task_dist 不能同时为空')

        batch_size = max(batch_sizes)

        if dataset_embed is not None:
            if dataset_embed.size(0) == 1 and batch_size > 1:
                dataset_embed = dataset_embed.expand(batch_size, -1)
            elif dataset_embed.size(0) != batch_size:
                raise ValueError('dataset_embed 与 dataset_task_dist 的 batch 维不兼容')

        if dataset_task_dist is not None:
            if dataset_task_dist.size(0) == 1 and batch_size > 1:
                dataset_task_dist = dataset_task_dist.expand(batch_size, -1)
            elif dataset_task_dist.size(0) != batch_size:
                raise ValueError('dataset_task_dist 与 dataset_embed 的 batch 维不兼容')

        return dataset_embed, dataset_task_dist

    def _forward_merge_head(
        self,
        dataset_embed: torch.Tensor,
        dataset_task_dist: torch.Tensor,
    ) -> torch.Tensor:
        dataset_embed, dataset_task_dist = self._prepare_aggregated_inputs(
            dataset_embed,
            dataset_task_dist,
        )

        if self.variant == ROUTER_VARIANT_HYBRID_DUAL:
            embed_features = self.embed_branch(dataset_embed)
            task_features = self.task_branch(dataset_task_dist)
            fusion_input = torch.cat([embed_features, task_features], dim=-1)
            alphas = self.fusion_head(fusion_input)
        else:
            features = []
            if self.use_dataset_embed:
                features.append(dataset_embed)
            if self.use_task_dist:
                features.append(dataset_task_dist)
            merge_input = torch.cat(features, dim=-1)
            alphas = self.merge_head(merge_input)

        return alphas.squeeze(0) if alphas.size(0) == 1 else alphas
    
    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码阶段：计算 per-sample embeddings 和 task logits。
        
        用于训练时的 forward()，也用于大数据集评估时的增量聚合。
        
        Args:
            input_ids: [Batch, Seq_Len] 输入 token ids
            attention_mask: [Batch, Seq_Len] attention mask (1=有效, 0=padding)
        
        Returns:
            sample_embeds: [Batch, Hidden] 每条样本的嵌入表示
            task_logits: [Batch, num_tasks] 每条样本的任务分类 logits
        """
        # 1. 共享 Embedding（冻结，不回传梯度到 Embedding）
        with torch.no_grad():
            embeddings = self.embedding(input_ids)  # [Batch, Seq, Hidden]
        
        # detach 确保梯度不回传到 Embedding
        embeddings = embeddings.detach()
        
        # 2. Per-sample Mean Pooling（考虑 attention_mask，排除 padding）
        mask = attention_mask.unsqueeze(-1).float()              # [Batch, Seq, 1]
        masked_embeds = embeddings * mask                        # [Batch, Seq, Hidden]
        sum_embeds = masked_embeds.sum(dim=1)                    # [Batch, Hidden]
        length = mask.sum(dim=1).clamp(min=1)                    # [Batch, 1] 避免除 0
        sample_embeds = sum_embeds / length                      # [Batch, Hidden]
        
        # 3. Task Classifier：Per-sample → 每条样本的任务 logits
        task_logits = self.task_classifier(sample_embeds)        # [Batch, num_tasks]
        
        return sample_embeds, task_logits
    
    def compute_alpha(
        self,
        sample_embeds: torch.Tensor,
        task_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        从 per-sample 特征聚合计算数据集级别 α。
        
        Args:
            sample_embeds: [N, Hidden] 来自一个或多个 batch 的样本嵌入
            task_logits: [N, num_tasks] 对应的任务分类 logits
        
        Returns:
            alphas: [num_tasks] 数据集级别的合并权重
        """
        dataset_embed = None
        dataset_task_dist = None
        if self.use_dataset_embed:
            dataset_embed = sample_embeds.mean(dim=0, keepdim=True)
        if self.use_task_dist:
            task_probs = F.softmax(task_logits, dim=-1).detach()
            dataset_task_dist = task_probs.mean(dim=0, keepdim=True)

        return self._forward_merge_head(dataset_embed, dataset_task_dist)
    
    def compute_alpha_from_aggregated(
        self,
        dataset_embed: torch.Tensor,
        dataset_task_dist: torch.Tensor,
    ) -> torch.Tensor:
        """
        从已聚合的数据集级特征计算 α。
        
        Args:
            dataset_embed: [1, Hidden] 或 [B, Hidden] 数据集平均嵌入
            dataset_task_dist: [1, num_tasks] 或 [B, num_tasks] 数据集任务分布
        
        Returns:
            alphas: [num_tasks] 或 [B, num_tasks] 数据集级别的合并权重
        """
        return self._forward_merge_head(dataset_embed, dataset_task_dist)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播：encode + 数据集级聚合 + merge_head → α。
        
        训练时 batch 模拟小数据集，部署时整个用户数据集作为输入。
        
        Args:
            input_ids: [Batch, Seq_Len] 输入 token ids
            attention_mask: [Batch, Seq_Len] attention mask (1=有效, 0=padding)
        
        Returns:
            alphas: [num_tasks] 数据集级合并权重
            task_logits: [Batch, num_tasks] 每条样本的任务分类 logits
        """
        sample_embeds, task_logits = self.encode(input_ids, attention_mask)
        alphas = self.compute_alpha(sample_embeds, task_logits)
        return alphas, task_logits
    
    def get_trainable_params(self):
        """返回所有可训练参数（两个 Head）"""
        params = []
        seen = set()
        for module in self._trainable_modules:
            for param in module.parameters():
                if id(param) in seen:
                    continue
                params.append(param)
                seen.add(id(param))
        return params
    
    def count_trainable_params(self) -> int:
        """统计可训练参数数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def count_total_params(self) -> int:
        """统计总参数数量（含冻结的 Embedding）"""
        return sum(p.numel() for p in self.parameters())


def create_router(base_model, variant: str = DEFAULT_ROUTER_VARIANT) -> Router:
    """
    从已加载的 GPTNeoForCausalLM 创建 Router。
    
    Args:
        base_model: GPTNeoForCausalLM 实例（已 from_pretrained 加载）
    
    Returns:
        Router 实例
    """
    embedding_layer = base_model.transformer.wte
    return Router(
        embedding_layer=embedding_layer,
        hidden_size=HIDDEN_SIZE,
        num_tasks=NUM_TASKS,
        variant=variant,
    )
