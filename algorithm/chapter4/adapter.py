"""
第四章算法适配层 — 统一适配器

封装 GPT-Neo 125M + Router + Task Vectors + 分类头的完整推理流程。
提供 4 个核心 API：
  - load_resources()        → 加载所有模型组件
  - get_baseline()          → 返回 4 任务基线准确率
  - infer_router(records)   → Router 推理, 返回 α 系数
  - merge_and_evaluate(records, alphas) → 合并 + 评测
"""
import os
import csv
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict
from transformers import GPTNeoForCausalLM, GPT2Tokenizer

from .config import (
    GPTNEO_PATH, TASK_VECTORS_DIR, HEADS_DIR,
    ROUTER_CHECKPOINT, META_PATH,
    TASK_NAMES, NUM_TASKS, HIDDEN_SIZE,
    TASK_CONFIGS, TASK_NAME_TO_IDX, IDX_TO_TASK_NAME,
    BASELINE_ACC,
)

logger = logging.getLogger(__name__)

# ── 分类头构建 (与 task_vectors.py 一致) ──

def _build_classification_head(hidden_size, num_classes):
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.LayerNorm(hidden_size),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_size, hidden_size // 2),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(hidden_size // 2, num_classes),
    )


# ── 全局缓存 ──
_cache = {
    'loaded': False,
    'device': None,
    'base_model': None,
    'base_params': None,
    'task_vectors': None,
    'heads': None,
    'router': None,
    'tokenizer': None,
    'meta': None,
}


def load_resources(device='cuda'):
    """加载所有模型组件到指定设备。

    Returns:
        dict: {
            status: 'ok',
            base_model_params: int,
            router_params: int,
            task_vector_tasks: list[str],
            head_tasks: list[str],
        }
    """
    if _cache['loaded'] and _cache['device'] == device:
        return _build_status()

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')
    logger.info(f"[Ch4] Loading resources to {dev}...")

    # 1. Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(GPTNEO_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    _cache['tokenizer'] = tokenizer

    # 2. Base model (GPT-Neo 125M)
    logger.info(f"[Ch4] Loading GPT-Neo 125M from {GPTNEO_PATH}")
    base_model = GPTNeoForCausalLM.from_pretrained(GPTNEO_PATH)
    base_model.config.output_hidden_states = True
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False
    base_model.to(dev)
    _cache['base_model'] = base_model

    base_params = {name: param.data for name, param in base_model.named_parameters()}
    _cache['base_params'] = base_params

    # 3. Task vectors
    logger.info("[Ch4] Loading task vectors...")
    task_vectors = {}
    for task_name in TASK_NAMES:
        tv_path = os.path.join(TASK_VECTORS_DIR, f'{task_name}.pt')
        task_vectors[task_name] = torch.load(tv_path, map_location=dev, weights_only=True)
        logger.info(f"  {task_name}: {len(task_vectors[task_name])} layers")
    _cache['task_vectors'] = task_vectors

    # 4. Classification heads
    logger.info("[Ch4] Loading classification heads...")
    heads = {}
    for task_name in TASK_NAMES:
        head_path = os.path.join(HEADS_DIR, f'{task_name}.pt')
        num_classes = TASK_CONFIGS[task_name]['num_classes']
        head = _build_classification_head(HIDDEN_SIZE, num_classes)
        head.load_state_dict(torch.load(head_path, map_location='cpu', weights_only=True))
        head.eval()
        for p in head.parameters():
            p.requires_grad = False
        head.to(dev)
        heads[task_name] = head
    _cache['heads'] = heads

    # 5. Router
    logger.info(f"[Ch4] Loading Router from {ROUTER_CHECKPOINT}")
    from .libs.router import Router, create_router, infer_router_variant_from_state_dict
    checkpoint = torch.load(ROUTER_CHECKPOINT, map_location='cpu', weights_only=False)
    router_state = checkpoint['router_state_dict']
    variant = infer_router_variant_from_state_dict(router_state)
    logger.info(f"  Router variant: {variant}")
    router = create_router(base_model, variant=variant)
    router.load_state_dict(router_state, strict=False)
    router.eval()
    router.to(dev)
    _cache['router'] = router

    # 6. Meta
    if os.path.exists(META_PATH):
        _cache['meta'] = torch.load(META_PATH, map_location='cpu', weights_only=False)

    _cache['device'] = device
    _cache['loaded'] = True
    logger.info("[Ch4] All resources loaded successfully.")
    return _build_status()


def _build_status():
    base_params_count = sum(p.numel() for p in _cache['base_model'].parameters())
    router_params_count = _cache['router'].count_trainable_params()
    return {
        'status': 'ok',
        'base_model_params': base_params_count,
        'router_params': router_params_count,
        'router_variant': _cache['router'].variant,
        'task_vector_tasks': list(_cache['task_vectors'].keys()),
        'head_tasks': list(_cache['heads'].keys()),
    }


def get_baseline():
    """返回 4 个单任务基线准确率。

    Returns:
        dict: {code: 0.8390, langid: 0.9173, law: 0.7057, math: 0.9585}
    """
    return dict(BASELINE_ACC)


# ── 数据处理 ──

def parse_uploaded_data(file_path):
    """解析用户上传的 CSV/JSON 数据。

    格式要求: text + label + task_id
    task_id: 0=code, 1=langid, 2=law, 3=math

    Returns:
        list[dict]: [{text, label, task_id, task_name}, ...]
    """
    ext = os.path.splitext(file_path)[1].lower()
    records = []

    if ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        for item in raw:
            records.append({
                'text': str(item['text']),
                'label': int(item['label']),
                'task_id': int(item['task_id']),
                'task_name': IDX_TO_TASK_NAME[int(item['task_id'])],
            })
    elif ext == '.csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tid = int(row['task_id'])
                records.append({
                    'text': str(row['text']),
                    'label': int(row['label']),
                    'task_id': tid,
                    'task_name': IDX_TO_TASK_NAME[tid],
                })
    else:
        raise ValueError(f"Unsupported file format: {ext}. Use .csv or .json")

    return records


def _tokenize_records(records, max_length=512):
    """Tokenize records into tensors."""
    tokenizer = _cache['tokenizer']
    texts = [r['text'] for r in records]
    encodings = tokenizer(
        texts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='pt',
    )
    return encodings


@torch.no_grad()
def infer_router(records, batch_size=32):
    """Router 推理：分析输入数据分布，输出 α 系数。

    Args:
        records: list[dict] — 解析后的数据记录
        batch_size: 推理批大小

    Returns:
        dict: {
            alphas: list[float],           # 4 维合并权重
            task_distribution: dict,        # 各任务的样本分布
            per_sample_task_preds: list,    # 每条样本的任务预测
            task_classification_acc: float, # 任务分类准确率
        }
    """
    _ensure_loaded()
    router = _cache['router']
    dev = torch.device(_cache['device'] if torch.cuda.is_available() else 'cpu')

    encodings = _tokenize_records(records)
    n = len(records)

    # Incremental aggregation
    embed_sum = torch.zeros(HIDDEN_SIZE, device=dev)
    task_prob_sum = torch.zeros(NUM_TASKS, device=dev)
    all_task_preds = []
    all_sample_embeds = []
    all_task_logits = []
    true_task_ids = [r['task_id'] for r in records]

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        input_ids = encodings['input_ids'][start:end].to(dev)
        attention_mask = encodings['attention_mask'][start:end].to(dev)

        sample_embeds, task_logits = router.encode(input_ids, attention_mask)
        embed_sum += sample_embeds.sum(dim=0)
        task_probs = F.softmax(task_logits, dim=-1)
        task_prob_sum += task_probs.sum(dim=0)
        all_task_preds.extend(task_logits.argmax(dim=-1).cpu().tolist())
        all_sample_embeds.append(sample_embeds)
        all_task_logits.append(task_logits)

    # Dataset-level α
    dataset_embed = (embed_sum / n).unsqueeze(0)
    dataset_task_dist = (task_prob_sum / n).unsqueeze(0)
    alphas = router.compute_alpha_from_aggregated(dataset_embed, dataset_task_dist)
    alphas_list = alphas.cpu().numpy().tolist()
    if isinstance(alphas_list[0], list):
        alphas_list = alphas_list[0]

    # Task distribution
    task_dist = {}
    for name in TASK_NAMES:
        tid = TASK_NAME_TO_IDX[name]
        count = sum(1 for r in records if r['task_id'] == tid)
        task_dist[name] = count

    # Task classification accuracy
    correct = sum(1 for pred, true in zip(all_task_preds, true_task_ids) if pred == true)
    task_cls_acc = correct / max(n, 1)

    # Per-task recall
    per_task_recall = {}
    for name in TASK_NAMES:
        tid = TASK_NAME_TO_IDX[name]
        tp = sum(1 for pred, true in zip(all_task_preds, true_task_ids) if pred == tid and true == tid)
        total = sum(1 for true in true_task_ids if true == tid)
        per_task_recall[name] = tp / max(total, 1)

    return {
        'alphas': alphas_list,
        'task_distribution': task_dist,
        'per_sample_task_preds': all_task_preds,
        'task_classification_acc': task_cls_acc,
        'per_task_recall': per_task_recall,
        '_sample_embeds': all_sample_embeds,
        '_task_logits': all_task_logits,
    }


@torch.no_grad()
def merge_and_evaluate(records, alphas=None, batch_size=16):
    """合并模型并评测全部任务。

    Args:
        records: list[dict] — 解析后的数据记录
        alphas: list[float] 或 None — 若 None 则先运行 Router 推理
        batch_size: 推理批大小

    Returns:
        dict: {
            alphas: list[float],
            per_task_acc: dict,          # {task_name: accuracy}
            per_task_samples: dict,      # {task_name: sample_count}
            overall_acc: float,
            baseline: dict,
        }
    """
    _ensure_loaded()
    dev = torch.device(_cache['device'] if torch.cuda.is_available() else 'cpu')

    # Router inference if alphas not provided
    if alphas is None:
        router_result = infer_router(records)
        alphas = router_result['alphas']

    alphas_tensor = torch.tensor(alphas, dtype=torch.float32, device=dev)

    # Compute merged params
    from .libs.merge import compute_merged_params, get_hidden_states, postprocess_hidden_states
    merged_params = compute_merged_params(
        _cache['base_params'], _cache['task_vectors'], alphas_tensor
    )

    # Tokenize
    encodings = _tokenize_records(records)
    n = len(records)

    # Group by task
    task_groups = {}
    for i, r in enumerate(records):
        tname = r['task_name']
        if tname not in task_groups:
            task_groups[tname] = {'indices': [], 'labels': []}
        task_groups[tname]['indices'].append(i)
        task_groups[tname]['labels'].append(r['label'])

    # Evaluate each task group
    per_task_acc = {}
    per_task_samples = {}
    total_correct = 0
    total_samples = 0

    for task_name, group in task_groups.items():
        if task_name not in _cache['heads']:
            logger.warning(f"No head for task {task_name}, skipping")
            continue

        head = _cache['heads'][task_name]
        indices = group['indices']
        labels = group['labels']

        correct = 0
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            batch_labels = labels[start:end]

            input_ids = encodings['input_ids'][batch_idx].to(dev)
            attention_mask = encodings['attention_mask'][batch_idx].to(dev)

            # Forward through merged model
            hidden = get_hidden_states(
                _cache['base_model'], merged_params, input_ids, attention_mask
            )
            sentence_repr = postprocess_hidden_states(hidden)
            logits = head(sentence_repr)
            preds = logits.argmax(dim=-1).cpu().tolist()

            label_tensor = torch.tensor(batch_labels, dtype=torch.long)
            pred_tensor = torch.tensor(preds, dtype=torch.long)
            correct += (pred_tensor == label_tensor).sum().item()

        acc = correct / max(len(indices), 1)
        per_task_acc[task_name] = round(acc, 4)
        per_task_samples[task_name] = len(indices)
        total_correct += correct
        total_samples += len(indices)

    overall_acc = total_correct / max(total_samples, 1)

    return {
        'alphas': alphas,
        'per_task_acc': per_task_acc,
        'per_task_samples': per_task_samples,
        'overall_acc': round(overall_acc, 4),
        'baseline': get_baseline(),
    }


@torch.no_grad()
def infer_and_evaluate_per_group(records, batch_size_router=32, batch_size_eval=16):
    """分组评测：先分类，再按预测任务分组分别计算 α 和评测。

    流程:
      1. Router task classifier 对每条样本预测所属任务
      2. 按预测任务将样本分组
      3. 对每组样本单独运行 Router 产生该组的 α
      4. 用该组 α 合并模型，用对应 head 评测该组
      5. 汇总：报告的 α = 各组 α 的平均值

    Returns:
        dict: 与原 evaluate API 返回格式相同
    """
    _ensure_loaded()
    router = _cache['router']
    dev = torch.device(_cache['device'] if torch.cuda.is_available() else 'cpu')

    from .libs.merge import compute_merged_params, get_hidden_states, postprocess_hidden_states

    encodings = _tokenize_records(records)
    n = len(records)

    # ── Step 1: 全量 Router encode → per-sample task prediction ──
    all_task_preds = []
    all_sample_embeds = []
    all_task_logits = []
    true_task_ids = [r['task_id'] for r in records]

    for start in range(0, n, batch_size_router):
        end = min(start + batch_size_router, n)
        input_ids = encodings['input_ids'][start:end].to(dev)
        attention_mask = encodings['attention_mask'][start:end].to(dev)
        sample_embeds, task_logits = router.encode(input_ids, attention_mask)
        all_task_preds.extend(task_logits.argmax(dim=-1).cpu().tolist())
        all_sample_embeds.append(sample_embeds)
        all_task_logits.append(task_logits)

    all_sample_embeds = torch.cat(all_sample_embeds, dim=0)  # [N, Hidden]
    all_task_logits = torch.cat(all_task_logits, dim=0)      # [N, num_tasks]

    # ── Step 2: 按 predicted task 分组 ──
    pred_groups = {}  # task_name → list of sample indices
    for i, pred_tid in enumerate(all_task_preds):
        tname = IDX_TO_TASK_NAME.get(pred_tid)
        if tname is None:
            continue
        if tname not in pred_groups:
            pred_groups[tname] = []
        pred_groups[tname].append(i)

    # Task classification accuracy
    correct_cls = sum(1 for pred, true in zip(all_task_preds, true_task_ids) if pred == true)
    task_cls_acc = correct_cls / max(n, 1)

    # Per-task recall
    per_task_recall = {}
    for name in TASK_NAMES:
        tid = TASK_NAME_TO_IDX[name]
        tp = sum(1 for pred, true in zip(all_task_preds, true_task_ids) if pred == tid and true == tid)
        total = sum(1 for true in true_task_ids if true == tid)
        per_task_recall[name] = tp / max(total, 1)

    # Task distribution (ground truth)
    task_dist = {}
    for name in TASK_NAMES:
        tid = TASK_NAME_TO_IDX[name]
        task_dist[name] = sum(1 for r in records if r['task_id'] == tid)

    # ── Step 3 & 4: 每组独立计算 α → 合并 → 评测 ──
    per_task_acc = {}
    per_task_samples = {}
    group_alphas = {}
    total_correct = 0
    total_samples = 0

    for task_name in TASK_NAMES:
        indices = pred_groups.get(task_name, [])
        if not indices:
            per_task_acc[task_name] = 0.0
            per_task_samples[task_name] = 0
            continue

        # 该组的 embeddings / logits → 组级 α
        group_embeds = all_sample_embeds[indices]     # [G, Hidden]
        group_logits = all_task_logits[indices]        # [G, num_tasks]
        group_alpha = router.compute_alpha(group_embeds, group_logits)
        group_alpha_list = group_alpha.cpu().numpy().tolist()
        if isinstance(group_alpha_list[0], list):
            group_alpha_list = group_alpha_list[0]
        group_alphas[task_name] = group_alpha_list
        group_alpha_tensor = torch.tensor(group_alpha_list, dtype=torch.float32, device=dev)

        logger.info(f"  [{task_name}] {len(indices)} samples, "
                     f"α={[round(a, 4) for a in group_alpha_list]}")

        # 合并模型
        merged_params = compute_merged_params(
            _cache['base_params'], _cache['task_vectors'], group_alpha_tensor
        )

        # 用 ground-truth label 评测（按原始 record 的 label）
        head = _cache['heads'].get(task_name)
        if head is None:
            logger.warning(f"No head for task {task_name}, skipping eval")
            continue

        group_correct = 0
        for start in range(0, len(indices), batch_size_eval):
            end = min(start + batch_size_eval, len(indices))
            batch_idx = indices[start:end]
            batch_labels = [records[i]['label'] for i in batch_idx]

            input_ids = encodings['input_ids'][batch_idx].to(dev)
            attention_mask = encodings['attention_mask'][batch_idx].to(dev)

            hidden = get_hidden_states(
                _cache['base_model'], merged_params, input_ids, attention_mask
            )
            sentence_repr = postprocess_hidden_states(hidden)
            logits = head(sentence_repr)
            preds = logits.argmax(dim=-1).cpu().tolist()

            label_tensor = torch.tensor(batch_labels, dtype=torch.long)
            pred_tensor = torch.tensor(preds, dtype=torch.long)
            group_correct += (pred_tensor == label_tensor).sum().item()

        acc = group_correct / max(len(indices), 1)
        per_task_acc[task_name] = round(acc, 4)
        per_task_samples[task_name] = len(indices)
        total_correct += group_correct
        total_samples += len(indices)

    overall_acc = total_correct / max(total_samples, 1)

    # ── Step 5: 报告 α = 各组 α 的平均 ──
    avg_alphas = [0.0] * NUM_TASKS
    num_groups = max(len(group_alphas), 1)
    for ga in group_alphas.values():
        for j in range(NUM_TASKS):
            avg_alphas[j] += ga[j]
    avg_alphas = [a / num_groups for a in avg_alphas]

    # ── Step 6: 归一化准确率 (merged / baseline) ──
    baseline = get_baseline()
    per_task_norm_acc = {}
    for task_name, acc in per_task_acc.items():
        bl = baseline.get(task_name, 1.0)
        per_task_norm_acc[task_name] = round(acc / bl, 4) if bl > 0 else 0.0
    norm_values = list(per_task_norm_acc.values())
    overall_norm_acc = round(sum(norm_values) / max(len(norm_values), 1), 4)

    return {
        'alphas': avg_alphas,
        'task_distribution': task_dist,
        'per_sample_task_preds': all_task_preds,
        'task_classification_acc': task_cls_acc,
        'per_task_recall': per_task_recall,
        'per_task_acc': per_task_acc,
        'per_task_norm_acc': per_task_norm_acc,
        'per_task_samples': per_task_samples,
        'overall_acc': round(overall_acc, 4),
        'overall_norm_acc': overall_norm_acc,
        'baseline': baseline,
    }


def _ensure_loaded():
    if not _cache['loaded']:
        load_resources()
