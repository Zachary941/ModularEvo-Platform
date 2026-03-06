"""
第三章算法适配层 — 路径配置

所有路径均指向 demo_system/data/ 下的本地副本，不依赖外部文件夹。
"""
import os

# ── 根目录 ──
# chapter3/ -> algorithm/ -> demo_system/
DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.path.join(DEMO_ROOT, 'data')
MODEL_ROOT = os.path.join(DATA_ROOT, 'models')
DATASET_ROOT = os.path.join(DATA_ROOT, 'datasets')

# ── 预训练基座 ──
CODEBERT_PATH = os.path.join(MODEL_ROOT, 'codebert-base')

# ── 模块化 mask 路径 ──
MODULE_PATHS = {
    'java':   os.path.join(MODEL_ROOT, 'module_java'),
    'python': os.path.join(MODEL_ROOT, 'module_python'),
}

# ── 微调模型 checkpoint 路径 ──
FINETUNED_PATHS = {
    'clone_detection': os.path.join(MODEL_ROOT, 'finetuned_clone', 'model.bin'),
    'code_search':     os.path.join(MODEL_ROOT, 'finetuned_search', 'pytorch_model.bin'),
}

# ── 评测数据路径 ──
EVAL_DATA_PATHS = {
    'clone_detection': os.path.join(DATASET_ROOT, 'clone_detection', 'test.txt'),
    'code_search':     os.path.join(DATASET_ROOT, 'code_search', 'cosqa_dev.json'),
}

# 克隆检测数据附属文件
CLONE_DATA_JSONL = os.path.join(DATASET_ROOT, 'clone_detection', 'data.jsonl')

# ── 合并方法名映射 (前端显示名 → MergingMethod 内部名) ──
MERGE_METHODS = {
    'task_arithmetic': 'task_arithmetic',
    'ties':            'ties_merging',
    'dare':            'mask_merging',
}
