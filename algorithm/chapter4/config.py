"""
第四章算法适配层 — 路径配置

所有路径均指向 demo_system/data/ 下的本地副本，不依赖外部文件夹。
"""
import os

# ── 根目录 ──
DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_ROOT = os.path.join(DEMO_ROOT, 'data')
MODEL_ROOT = os.path.join(DATA_ROOT, 'models')
DATASET_ROOT = os.path.join(DATA_ROOT, 'datasets')

# ── GPT-Neo 125M 基座 ──
GPTNEO_PATH = os.path.join(MODEL_ROOT, 'gpt-neo-125m')

# ── 预计算资源路径 ──
TASK_VECTORS_DIR = os.path.join(MODEL_ROOT, 'task_vectors')
HEADS_DIR = os.path.join(MODEL_ROOT, 'heads')
ROUTER_CHECKPOINT = os.path.join(MODEL_ROOT, 'router_checkpoint', 'best_router.pt')
META_PATH = os.path.join(MODEL_ROOT, 'router_meta.pt')

# ── 示例数据集 ──
SAMPLE_DATASET_PATH = os.path.join(DATASET_ROOT, 'sample_datasets', 'mixed_sample_50.csv')

# ── 任务配置 ──
TASK_NAMES = ['code', 'langid', 'law', 'math']
NUM_TASKS = len(TASK_NAMES)
HIDDEN_SIZE = 768

TASK_CONFIGS = {
    'code':   {'num_classes': 1006, 'description': '代码语言分类 (1006种编程语言)'},
    'langid': {'num_classes': 6,    'description': '欧洲语言识别 (6种语言)'},
    'law':    {'num_classes': 13,   'description': '法律分类/SCOTUS (13个类别)'},
    'math':   {'num_classes': 25,   'description': '数学QA分类 (25个topic)'},
}

TASK_NAME_TO_IDX = {name: idx for idx, name in enumerate(TASK_NAMES)}
IDX_TO_TASK_NAME = {idx: name for idx, name in enumerate(TASK_NAMES)}

# ── 基线准确率 (微调模型在各任务测试集上的 ACC) ──
BASELINE_ACC = {
    'code':   0.8390,
    'langid': 0.9173,
    'law':    0.7057,
    'math':   0.9585,
}
