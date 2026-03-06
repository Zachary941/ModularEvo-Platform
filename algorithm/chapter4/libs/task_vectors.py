"""
步骤1：预计算与加载 Task Vectors（任务向量）和分类头。

功能：
1. 加载基座模型 GPTNeoForCausalLM 的权重作为 θ_base
2. 依次加载4个微调模型的 state_dict
3. 提取 Backbone 部分（base_model.* 前缀）计算 Task Vector：τ_i = θ_finetuned_i - θ_base
4. 提取分类头部分（classification_head.* 前缀）单独保存
5. 将 Task Vectors 和分类头保存到磁盘

输出文件（保存到 router/data/）：
  - task_vectors/{task_name}.pt  — 各任务的 Task Vector (Dict[str, Tensor])
  - heads/{task_name}.pt         — 各任务的分类头权重 (OrderedDict)
  - meta.pt                      — 元信息（任务名、num_classes、参数数量统计等）
"""

import os
import sys
import torch
import torch.nn as nn
from collections import OrderedDict
import logging
import time
import json

# 将 router 目录加入 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from .config import (
    BASE_MODEL_PATH, HIDDEN_SIZE, TASK_NAMES, TASK_CONFIGS,
    TASK_VECTORS_DIR, HEADS_DIR, DATA_DIR
)

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def build_classification_head(hidden_size: int, num_classes: int) -> nn.Sequential:
    """
    构建与微调代码中完全一致的分类头结构。
    """
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),        # 0
        nn.LayerNorm(hidden_size),                   # 1
        nn.ReLU(),                                   # 2
        nn.Dropout(0.2),                             # 3
        nn.Linear(hidden_size, hidden_size // 2),    # 4
        nn.ReLU(),                                   # 5
        nn.Dropout(0.1),                             # 6
        nn.Linear(hidden_size // 2, num_classes)     # 7
    )


def compute_and_save_task_vectors():
    """
    主函数：计算所有任务的 Task Vector 和分类头，保存到磁盘。
    """
    start_time = time.time()

    # ===== 创建输出目录 =====
    os.makedirs(TASK_VECTORS_DIR, exist_ok=True)
    os.makedirs(HEADS_DIR, exist_ok=True)
    logger.info(f"输出目录: {DATA_DIR}")

    # ===== 1. 加载基座模型 =====
    logger.info(f"正在加载基座模型: {BASE_MODEL_PATH}")
    from transformers import GPTNeoForCausalLM
    base_model = GPTNeoForCausalLM.from_pretrained(BASE_MODEL_PATH)
    base_model.config.output_hidden_states = True

    # 提取基座模型参数（不含 base_model. 前缀）
    base_params = {}
    for name, param in base_model.named_parameters():
        base_params[name] = param.data.clone().cpu()
        param.requires_grad = False

    total_base_params = sum(p.numel() for p in base_params.values())
    logger.info(f"基座模型参数数量: {total_base_params:,} ({total_base_params/1e6:.2f}M)")
    logger.info(f"基座模型参数名示例: {list(base_params.keys())[:5]}")

    # ===== 2. 逐个任务提取 Task Vector 和分类头 =====
    meta_info = {
        'base_model_path': BASE_MODEL_PATH,
        'hidden_size': HIDDEN_SIZE,
        'task_names': TASK_NAMES,
        'tasks': {},
    }

    for task_name in TASK_NAMES:
        config = TASK_CONFIGS[task_name]
        model_path = config['model_path']
        num_classes = config['num_classes']

        logger.info(f"\n{'='*60}")
        logger.info(f"处理任务: {task_name} ({config['description']})")
        logger.info(f"  模型路径: {model_path}")
        logger.info(f"  分类数: {num_classes}")

        # 检查模型文件是否存在
        if not os.path.exists(model_path):
            logger.error(f"  模型文件不存在: {model_path}")
            continue

        # 加载微调模型的 state_dict
        logger.info(f"  正在加载 state_dict...")
        state_dict = torch.load(model_path, map_location='cpu')
        logger.info(f"  state_dict 总键数: {len(state_dict)}")

        # ---------- 2a. 提取 Backbone 部分并计算 Task Vector ----------
        tau = OrderedDict()
        backbone_keys_matched = 0
        backbone_keys_missed = 0
        nonzero_params = 0
        total_params = 0
        max_diff = 0.0

        for key, base_param in base_params.items():
            finetuned_key = f"base_model.{key}"
            if finetuned_key in state_dict:
                diff = state_dict[finetuned_key].cpu() - base_param
                tau[key] = diff
                backbone_keys_matched += 1

                # 统计稀疏性
                n_elem = diff.numel()
                n_nonzero = (diff.abs() > 1e-7).sum().item()
                total_params += n_elem
                nonzero_params += n_nonzero
                layer_max = diff.abs().max().item()
                if layer_max > max_diff:
                    max_diff = layer_max
            else:
                backbone_keys_missed += 1

        sparsity = 1.0 - (nonzero_params / total_params) if total_params > 0 else 0.0

        logger.info(f"  Task Vector 统计:")
        logger.info(f"    匹配的参数层数: {backbone_keys_matched}")
        logger.info(f"    未匹配的参数层数: {backbone_keys_missed}")
        logger.info(f"    总参数量: {total_params:,}")
        logger.info(f"    非零参数量: {nonzero_params:,} ({nonzero_params/total_params*100:.2f}%)")
        logger.info(f"    稀疏率: {sparsity*100:.2f}%")
        logger.info(f"    最大差值: {max_diff:.6f}")

        # 保存 Task Vector
        tv_path = os.path.join(TASK_VECTORS_DIR, f"{task_name}.pt")
        torch.save(tau, tv_path)
        logger.info(f"  Task Vector 已保存: {tv_path}")

        # ---------- 2b. 提取并保存分类头 ----------
        head_state = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('classification_head.'):
                # 去掉 "classification_head." 前缀，与 nn.Sequential 的 key 对齐
                new_key = k.replace('classification_head.', '')
                head_state[new_key] = v.cpu()

        logger.info(f"  分类头参数层数: {len(head_state)}")

        # 验证分类头可以正确加载
        head = build_classification_head(HIDDEN_SIZE, num_classes)
        try:
            head.load_state_dict(head_state)
            head_param_count = sum(p.numel() for p in head.parameters())
            logger.info(f"  分类头参数量: {head_param_count:,}")
            logger.info(f"  分类头加载验证: ✓ 成功")
        except Exception as e:
            logger.error(f"  分类头加载验证: ✗ 失败 - {e}")
            # 打印 key 对比以便调试
            logger.error(f"    期望 keys: {list(head.state_dict().keys())}")
            logger.error(f"    实际 keys: {list(head_state.keys())}")
            continue

        # 保存分类头权重
        head_path = os.path.join(HEADS_DIR, f"{task_name}.pt")
        torch.save(head_state, head_path)
        logger.info(f"  分类头已保存: {head_path}")

        # 记录元信息
        meta_info['tasks'][task_name] = {
            'num_classes': num_classes,
            'description': config['description'],
            'model_path': model_path,
            'tv_path': tv_path,
            'head_path': head_path,
            'backbone_keys_matched': backbone_keys_matched,
            'total_params': total_params,
            'nonzero_params': nonzero_params,
            'sparsity': sparsity,
            'max_diff': max_diff,
            'head_param_count': head_param_count,
        }

        # 释放内存
        del state_dict, tau, head_state, head
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ===== 3. 保存元信息 =====
    meta_path = os.path.join(DATA_DIR, 'meta.pt')
    torch.save(meta_info, meta_path)
    logger.info(f"\n元信息已保存: {meta_path}")

    # 同时保存一份 JSON 版（方便查看）
    meta_json_path = os.path.join(DATA_DIR, 'meta.json')
    # 转换为 JSON 可序列化格式
    meta_json = {
        'base_model_path': meta_info['base_model_path'],
        'hidden_size': meta_info['hidden_size'],
        'task_names': meta_info['task_names'],
        'tasks': {}
    }
    for t_name, t_info in meta_info['tasks'].items():
        meta_json['tasks'][t_name] = {
            'num_classes': t_info['num_classes'],
            'description': t_info['description'],
            'backbone_keys_matched': t_info['backbone_keys_matched'],
            'total_params': t_info['total_params'],
            'nonzero_params': t_info['nonzero_params'],
            'sparsity': round(t_info['sparsity'], 6),
            'max_diff': round(t_info['max_diff'], 6),
            'head_param_count': t_info['head_param_count'],
        }
    with open(meta_json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_json, f, indent=2, ensure_ascii=False)
    logger.info(f"元信息(JSON)已保存: {meta_json_path}")

    elapsed = time.time() - start_time
    logger.info(f"\n{'='*60}")
    logger.info(f"步骤1完成！总耗时: {elapsed:.1f}s")
    logger.info(f"输出文件:")
    logger.info(f"  Task Vectors: {TASK_VECTORS_DIR}/")
    logger.info(f"  分类头: {HEADS_DIR}/")
    logger.info(f"  元信息: {meta_path}")

    return meta_info


def load_task_vectors_and_heads():
    """
    从磁盘加载已保存的 Task Vectors 和分类头。
    供后续步骤（Router 训练、合并、推理）使用。
    
    Returns:
        base_model: GPTNeoForCausalLM — 基座模型（已冻结）
        base_params: Dict[str, Tensor] — 基座模型参数
        task_vectors: Dict[str, Dict[str, Tensor]] — 各任务的 Task Vector
        heads: Dict[str, nn.Sequential] — 各任务的分类头（已冻结）
    """
    from transformers import GPTNeoForCausalLM

    # 加载基座模型
    logger.info(f"加载基座模型: {BASE_MODEL_PATH}")
    base_model = GPTNeoForCausalLM.from_pretrained(BASE_MODEL_PATH)
    base_model.config.output_hidden_states = True
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    base_params = {name: param.data for name, param in base_model.named_parameters()}

    # 加载 Task Vectors
    task_vectors = {}
    for task_name in TASK_NAMES:
        tv_path = os.path.join(TASK_VECTORS_DIR, f"{task_name}.pt")
        if not os.path.exists(tv_path):
            raise FileNotFoundError(f"Task Vector 文件不存在: {tv_path}，请先运行 compute_and_save_task_vectors()")
        task_vectors[task_name] = torch.load(tv_path, map_location='cpu')
        logger.info(f"  已加载 Task Vector: {task_name} ({len(task_vectors[task_name])} 层)")

    # 加载分类头
    heads = {}
    for task_name in TASK_NAMES:
        head_path = os.path.join(HEADS_DIR, f"{task_name}.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"分类头文件不存在: {head_path}，请先运行 compute_and_save_task_vectors()")
        num_classes = TASK_CONFIGS[task_name]['num_classes']
        head = build_classification_head(HIDDEN_SIZE, num_classes)
        head.load_state_dict(torch.load(head_path, map_location='cpu'))
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        heads[task_name] = head
        logger.info(f"  已加载分类头: {task_name} (num_classes={num_classes})")

    return base_model, base_params, task_vectors, heads


if __name__ == '__main__':
    compute_and_save_task_vectors()
