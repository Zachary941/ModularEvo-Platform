"""
第三章：基于 Transformer 模型模块化与动态合并的算法适配层

对外暴露：
  - model_loader: load_base_model, load_sparse_module, load_finetuned_model, get_module_info
  - evaluator:    evaluate_task, evaluate_clone, evaluate_search
  - merger:       merge_models, merge_and_evaluate, apply_merged_params
  - config:       路径和常量
"""
from .model_loader import load_base_model, load_sparse_module, load_finetuned_model, get_module_info
from .evaluator import evaluate_task, evaluate_clone, evaluate_search
from .merger import merge_models, merge_and_evaluate, apply_merged_params
