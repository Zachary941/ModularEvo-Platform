"""
第三章算法适配层 — 评测统一入口

封装克隆检测和代码搜索两个下游任务的评测逻辑。
"""
import os
import tempfile
import multiprocessing

from .config import EVAL_DATA_PATHS


def evaluate_clone(model, tokenizer, data_path=None, output_dir=None):
    """克隆检测评测。

    Returns:
        dict: {eval_recall, eval_precision, eval_f1, eval_threshold}
    """
    from .libs.clone_model import evaluate as _eval_clone

    if data_path is None:
        data_path = EVAL_DATA_PATHS['clone_detection']
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='clone_eval_')

    pool = multiprocessing.Pool(min(16, os.cpu_count() or 4))
    try:
        result = _eval_clone(model, tokenizer, data_path, output_dir, pool=pool)
    finally:
        pool.close()
        pool.join()
    return result


def evaluate_search(model, tokenizer, data_path=None, output_dir=None):
    """代码搜索评测。

    Returns:
        dict: {acc, precision, recall, f1, acc_and_f1, eval_loss}
    """
    from .libs.search_model import evaluate as _eval_search

    if data_path is None:
        data_path = EVAL_DATA_PATHS['code_search']
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix='search_eval_')

    return _eval_search(model, tokenizer, data_path, output_dir)


_TASK_EVALUATORS = {
    'clone_detection': evaluate_clone,
    'code_search': evaluate_search,
}


def evaluate_task(task, model, tokenizer, **kwargs):
    """统一评测入口。

    Args:
        task: 'clone_detection' | 'code_search'
        model: 已加载的任务模型
        tokenizer: RobertaTokenizer
        **kwargs: 可选 data_path, output_dir

    Returns:
        dict: 评测结果
    """
    if task not in _TASK_EVALUATORS:
        raise ValueError(f"Unknown task: {task}. Available: {list(_TASK_EVALUATORS.keys())}")
    return _TASK_EVALUATORS[task](model, tokenizer, **kwargs)
