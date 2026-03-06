"""第三章 API — Pydantic 请求/响应模型"""
from pydantic import BaseModel
from typing import Optional, Dict, Any, List


# ── 模块相关 ──
class ModuleInfo(BaseModel):
    language: str
    path: str
    exists: bool
    wrr_label: str


class LayerStat(BaseModel):
    name: str
    total: int
    nonzero: int
    ratio: float


class LoadModuleRequest(BaseModel):
    language: str  # 'java' | 'python'


class LoadModuleResponse(BaseModel):
    language: str
    sparsity: float
    wrr: float
    layer_stats: List[LayerStat]


# ── 微调模型相关 ──
class FinetunedInfo(BaseModel):
    task: str
    params_m: float  # 参数量 (百万)
    checkpoint: str  # checkpoint 文件名


class EvalResult(BaseModel):
    task: str
    metrics: Dict[str, float]


# ── 合并相关 ──
class MergeRequest(BaseModel):
    method: str  # 'task_arithmetic' | 'ties' | 'dare'
    scaling_coefficients: List[float] = [0.5, 0.5]
    param_value_mask_rate: float = 0.8  # TIES 专用
    weight_mask_rates: List[float] = [0.5, 0.5]  # DARE 专用


class MergeResponse(BaseModel):
    method: str
    scaling_coefficients: List[float]
    results: Dict[str, Dict[str, float]]


# ── 知识图谱 ──
class GraphNode(BaseModel):
    id: str
    name: str
    type: str  # pretrained / module / finetuned / merged
    meta: Dict[str, Any] = {}


class GraphEdge(BaseModel):
    source: str
    target: str
    relation: str
    style: str = "solid"


class GraphResponse(BaseModel):
    nodes: List[GraphNode]
    edges: List[GraphEdge]
