"""第四章 API — Pydantic 请求/响应模型"""
from pydantic import BaseModel
from typing import Dict, Any, List, Optional


# ── 状态 ──
class StatusResponse(BaseModel):
    loaded: bool
    base_model_params: Optional[int] = None
    router_params: Optional[int] = None
    router_variant: Optional[str] = None
    task_vector_tasks: Optional[List[str]] = None
    head_tasks: Optional[List[str]] = None


# ── 基线 ──
class BaselineResponse(BaseModel):
    baseline: Dict[str, float]


# ── 模型信息 ──
class ModelInfo(BaseModel):
    id: str
    name: str
    type: str
    params: Optional[str] = None
    description: Optional[str] = None
    baseline_acc: Optional[float] = None
    num_classes: Optional[int] = None


# ── 数据集上传 ──
class UploadResponse(BaseModel):
    filename: str
    total_samples: int
    task_distribution: Dict[str, int]
    preview: List[Dict[str, Any]]


# ── 评测 ──
class EvaluateRequest(BaseModel):
    file_path: str  # 服务端保存的文件路径


class EvaluateResponse(BaseModel):
    alphas: List[float]
    task_distribution: Dict[str, int]
    task_classification_acc: float
    per_task_recall: Dict[str, float]
    per_task_acc: Dict[str, float]
    per_task_norm_acc: Dict[str, float]
    per_task_samples: Dict[str, int]
    overall_acc: float
    overall_norm_acc: float
    baseline: Dict[str, float]


# ── 知识图谱（复用 ch3 的 schema） ──
class GraphNode(BaseModel):
    id: str
    name: str
    type: str
    meta: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    style: str = "solid"


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
