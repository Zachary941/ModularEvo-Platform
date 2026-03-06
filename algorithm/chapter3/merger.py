"""
第三章算法适配层 — 模型合并统一入口

支持 task_arithmetic / ties_merging / mask_merging (DARE) 三种合并方法。
流程：加载微调模型 → 计算 task vector → 合并 encoder → 覆盖到各任务模型 → 评测。
"""
import copy
import torch

from .config import MERGE_METHODS, FINETUNED_PATHS
from .model_loader import load_base_model, load_finetuned_model
from .evaluator import evaluate_task
from .libs.merging_methods import MergingMethod

# 排除 task-specific head 参数
EXCLUDE_PARAM_REGEX = [".*classifier.*", ".*mlp.*"]


def merge_models(method_name, scaling_coefficients=None, device="cuda",
                 param_value_mask_rate=0.8, weight_mask_rates=None):
    """执行模型合并，返回合并后的 encoder 参数字典。

    Args:
        method_name: 'task_arithmetic' | 'ties' | 'dare'
        scaling_coefficients: list of float, 每个任务的缩放系数, 默认 [0.5, 0.5]
        device: 'cuda' | 'cpu'
        param_value_mask_rate: TIES 方法的 mask 比率
        weight_mask_rates: DARE 方法的每模型 mask 比率

    Returns:
        dict: {merged_params, base_model, config, tokenizer, finetuned_models}
    """
    if method_name not in MERGE_METHODS:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(MERGE_METHODS.keys())}")

    internal_name = MERGE_METHODS[method_name]
    if scaling_coefficients is None:
        scaling_coefficients = [0.5, 0.5]
    if weight_mask_rates is None:
        weight_mask_rates = [0.5, 0.5]

    base_model, config, tokenizer = load_base_model(device)

    # 加载微调模型（完整任务模型，包含 task head）
    ft_clone = load_finetuned_model('clone_detection', base_model, config, tokenizer, device)
    ft_search = load_finetuned_model('code_search', base_model, config, tokenizer, device)
    models_to_merge = [ft_clone, ft_search]

    # 构建预训练基座（未微调的包装模型，用于计算 task vector 基准）
    from .libs.clone_model import Model as CloneModel
    pretrained_model = CloneModel(copy.deepcopy(base_model), config, tokenizer).to(device)

    merging_method = MergingMethod(internal_name)

    merge_kwargs = {
        'merged_model': pretrained_model,
        'models_to_merge': models_to_merge,
        'exclude_param_names_regex': EXCLUDE_PARAM_REGEX,
        'scaling_coefficients': scaling_coefficients,
    }

    if internal_name == 'ties_merging':
        merge_kwargs['param_value_mask_rate'] = param_value_mask_rate
    elif internal_name == 'mask_merging':
        merge_kwargs['weight_mask_rates'] = weight_mask_rates
        merge_kwargs['mask_apply_method'] = 'task_arithmetic'
        merge_kwargs['weight_format'] = 'delta_weight'
        merge_kwargs['use_weight_rescale'] = True
        merge_kwargs['mask_strategy'] = 'random'

    merged_params = merging_method.get_merged_model(**merge_kwargs)

    return {
        'merged_params': merged_params,
        'base_model': base_model,
        'config': config,
        'tokenizer': tokenizer,
        'finetuned_models': {'clone_detection': ft_clone, 'code_search': ft_search},
    }


def apply_merged_params(task, merge_result, device="cuda"):
    """将合并后的 encoder 参数应用到指定任务模型。

    先加载微调参数（含 task head），再用合并参数覆盖 encoder。

    Returns:
        nn.Module: 合并后的任务模型
    """
    config = merge_result['config']
    tokenizer = merge_result['tokenizer']
    base_model = merge_result['base_model']
    merged_params = merge_result['merged_params']

    # 加载完整微调模型（含 task head）
    model = load_finetuned_model(task, base_model, config, tokenizer, device)
    # 覆盖 encoder 参数为合并结果
    model.load_state_dict(merged_params, strict=False)
    model.eval()
    return model


def merge_and_evaluate(method_name, tasks=None, scaling_coefficients=None,
                       device="cuda", **merge_kwargs):
    """合并 + 评测完整流程。

    Args:
        method_name: 'task_arithmetic' | 'ties' | 'dare'
        tasks: 要评测的任务列表, 默认全部
        scaling_coefficients: 缩放系数
        device: 'cuda' | 'cpu'

    Returns:
        dict: {method, scaling_coefficients, results: {task: eval_metrics}}
    """
    if tasks is None:
        tasks = ['clone_detection', 'code_search']

    merge_result = merge_models(method_name, scaling_coefficients=scaling_coefficients,
                                device=device, **merge_kwargs)

    results = {}
    for task in tasks:
        merged_model = apply_merged_params(task, merge_result, device)
        metrics = evaluate_task(task, merged_model, merge_result['tokenizer'])
        results[task] = metrics

    return {
        'method': method_name,
        'scaling_coefficients': scaling_coefficients or [0.5, 0.5],
        'results': results,
    }
