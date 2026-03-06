"""Demo 系统配置"""
import os
from pathlib import Path

# 项目根目录 (demo_system/)
PROJECT_ROOT = Path(os.environ.get(
    "DEMO_PROJECT_ROOT",
    Path(__file__).resolve().parent.parent.parent
))

# 算法代码根目录 (old_ModularEvo/)
ALGO_ROOT = Path(os.environ.get(
    "DEMO_ALGO_ROOT",
    PROJECT_ROOT.parent
))

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"
DB_PATH = PROJECT_ROOT / "backend" / "demo.db"

# API 配置
API_PREFIX = "/api"
CORS_ORIGINS = ["http://localhost:3000", "http://127.0.0.1:3000"]

# 数据集限制
MAX_UPLOAD_SAMPLES = 200
