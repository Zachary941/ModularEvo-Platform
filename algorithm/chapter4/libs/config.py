"""
全局配置文件：模型路径、任务定义、超参数等。
所有路径均指向 demo_system/data/models/ 下的本地副本。
"""
import os

# ===== 项目根目录 =====
# libs/ -> chapter4/ -> algorithm/ -> demo_system/
DEMO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
MODEL_ROOT = os.path.join(DEMO_ROOT, 'data', 'models')

# 保留兼容性
PROJECT_ROOT = DEMO_ROOT
ROUTER_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = MODEL_ROOT  # task_vectors/ 和 heads/ 在此目录下

# ===== 基座模型 =====
BASE_MODEL_PATH = os.path.join(MODEL_ROOT, 'gpt-neo-125m')
HIDDEN_SIZE = 768  # GPT-Neo 125M

# ===== 任务配置 =====
TASK_NAMES = ['code', 'langid', 'law', 'math']
NUM_TASKS = len(TASK_NAMES)

TASK_CONFIGS = {
    'code': {
        'num_classes': 1006,
        'model_path': None,  # 使用预计算的 task_vectors，不需要微调模型路径
        'description': '代码语言分类 (1006种编程语言)',
    },
    'langid': {
        'num_classes': 6,
        'model_path': None,
        'description': '欧洲语言识别 (6种语言)',
    },
    'law': {
        'num_classes': 13,
        'model_path': None,
        'description': '法律分类/SCOTUS (13个类别)',
    },
    'math': {
        'num_classes': 25,
        'model_path': None,
        'description': '数学QA分类 (25个topic)',
    },
}

# ===== Task Vector 和分类头存储路径 =====
TASK_VECTORS_DIR = os.path.join(DATA_DIR, 'task_vectors')
HEADS_DIR = os.path.join(DATA_DIR, 'heads')

# ===== 任务名到索引的映射 =====
TASK_NAME_TO_IDX = {name: idx for idx, name in enumerate(TASK_NAMES)}
IDX_TO_TASK_NAME = {idx: name for idx, name in enumerate(TASK_NAMES)}
