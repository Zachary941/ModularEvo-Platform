"""
第三章算法适配层 — 模型加载统一入口

提供 CodeBERT 基座、模块化 mask、微调模型的加载功能。
所有依赖均来自本地 libs/ 目录。
"""
import os
import copy
import torch
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

from .config import CODEBERT_PATH, MODULE_PATHS, FINETUNED_PATHS
from .libs.sparse_utils import Binarization


def load_base_model(device="cuda"):
    """加载 CodeBERT 基座模型。

    Returns:
        (model, config, tokenizer)
    """
    config = RobertaConfig.from_pretrained(CODEBERT_PATH)
    tokenizer = RobertaTokenizer.from_pretrained(CODEBERT_PATH)
    model = RobertaModel.from_pretrained(CODEBERT_PATH, config=config)
    model.to(device)
    model.eval()
    return model, config, tokenizer


def load_sparse_module(language, base_model=None, device="cuda"):
    """加载预训练 mask 并应用到基座模型，返回稀疏统计信息。

    Args:
        language: 'java' | 'python'
        base_model: 已加载的 RobertaModel，为 None 时自动加载
        device: 'cuda' | 'cpu'

    Returns:
        dict: {model, sparsity, wrr, layer_stats: [{name, total, nonzero, ratio}]}
    """
    if language not in MODULE_PATHS:
        raise ValueError(f"Unknown language: {language}. Available: {list(MODULE_PATHS.keys())}")

    module_path = os.path.join(MODULE_PATHS[language], 'pytorch_model.bin')
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Module mask not found: {module_path}")

    if base_model is None:
        base_model, _, _ = load_base_model(device)

    model = copy.deepcopy(base_model)
    module_state = torch.load(module_path, map_location=device, weights_only=False)
    model_state = model.state_dict()
    sparse_model_state = copy.deepcopy(model_state)

    prefix = 'roberta.'
    layer_stats = []
    total_params = 0
    non_zero_params = 0

    for k in model_state:
        tmp_k = f'{prefix}{k}'
        if tmp_k in module_state:
            if f'{tmp_k}_mask' in module_state:
                init_weight = module_state[tmp_k]
                weight_mask = Binarization.apply(module_state[f'{tmp_k}_mask'])
                masked_weight = init_weight * weight_mask

                layer_total = torch.numel(masked_weight)
                layer_nonzero = torch.count_nonzero(masked_weight).item()
                total_params += layer_total
                non_zero_params += layer_nonzero

                layer_stats.append({
                    'name': k,
                    'total': layer_total,
                    'nonzero': layer_nonzero,
                    'ratio': layer_nonzero / layer_total if layer_total > 0 else 0,
                })
                sparse_model_state[k] = masked_weight
            else:
                sparse_model_state[k] = module_state[tmp_k]

    model.load_state_dict(sparse_model_state)
    wrr = (non_zero_params / total_params * 100) if total_params > 0 else 0
    sparsity = 100 - wrr

    return {
        'model': model,
        'sparsity': round(sparsity, 2),
        'wrr': round(wrr, 2),
        'layer_stats': layer_stats,
    }


def load_finetuned_model(task, base_model=None, config=None, tokenizer=None, device="cuda"):
    """加载指定任务的微调模型。

    Args:
        task: 'clone_detection' | 'code_search'
        base_model, config, tokenizer: 已加载的基座，为 None 时自动加载
        device: 'cuda' | 'cpu'

    Returns:
        nn.Module: 加载了 checkpoint 的任务模型
    """
    if task not in FINETUNED_PATHS:
        raise ValueError(f"Unknown task: {task}. Available: {list(FINETUNED_PATHS.keys())}")

    ckpt_path = FINETUNED_PATHS[task]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Finetuned model not found: {ckpt_path}")

    if base_model is None or config is None or tokenizer is None:
        base_model, config, tokenizer = load_base_model(device)

    encoder = copy.deepcopy(base_model)

    if task == 'clone_detection':
        from .libs.clone_model import Model as CloneModel
        model = CloneModel(encoder, config, tokenizer)
    elif task == 'code_search':
        from .libs.search_model import Model as SearchModel
        model = SearchModel(encoder, config, tokenizer)

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_module_info(language):
    """获取模块元信息（不加载到 GPU）。

    Returns:
        dict: {language, path, exists, wrr_label}
    """
    if language not in MODULE_PATHS:
        raise ValueError(f"Unknown language: {language}")

    path = MODULE_PATHS[language]
    bin_path = os.path.join(path, 'pytorch_model.bin')

    wrr_labels = {
        'java': '22.94%',
        'python': '24.15%',
    }

    return {
        'language': language,
        'path': path,
        'exists': os.path.exists(bin_path),
        'wrr_label': wrr_labels.get(language, 'unknown'),
    }
