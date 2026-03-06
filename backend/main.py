"""TransModular Demo 系统 — FastAPI 入口"""
import sys
from pathlib import Path

# 确保 backend/ 在 sys.path 中
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.config import CORS_ORIGINS, API_PREFIX
from models.database import init_db
from models.schemas import HealthResponse

app = FastAPI(
    title="TransModular Demo",
    description="Transformer 模型模块化与动态合并展示系统",
    version="0.1.0",
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup():
    """应用启动时初始化数据库"""
    init_db()


@app.get("/health", response_model=HealthResponse)
def health_check():
    """健康检查接口"""
    return {"status": "ok", "version": "0.1.0"}


# ---- 后续 API 路由将在 P2/P4 阶段添加 ----
# from api import chapter3, chapter4, graph
# app.include_router(chapter3.router, prefix=f"{API_PREFIX}/ch3", tags=["Chapter 3"])
# app.include_router(chapter4.router, prefix=f"{API_PREFIX}/ch4", tags=["Chapter 4"])
