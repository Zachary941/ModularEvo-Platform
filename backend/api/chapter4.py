"""第四章 API 路由 — AutoRouter (GPT-Neo + Router 动态合并)

提供模型加载状态、基线准确率、知识图谱、数据集上传与评测接口。
"""
import os
import sys
import uuid
import shutil
import traceback
from pathlib import Path
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse

_demo_root = Path(__file__).resolve().parent.parent.parent
_workspace_root = _demo_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from .ch4_schemas import (
    StatusResponse, BaselineResponse, ModelInfo,
    UploadResponse, EvaluateRequest, EvaluateResponse,
    GraphResponse, GraphNode, GraphEdge,
)

router = APIRouter()

# ── 上传文件暂存目录 ──
_UPLOAD_DIR = _demo_root / "data" / "uploads" / "ch4"
_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# ── 示例数据集路径 ──
_SAMPLE_DATASET = _demo_root / "data" / "datasets" / "sample_datasets" / "mixed_sample_50.csv"


# ────────────────────────────────────────
# GET /status — 模型加载状态
# ────────────────────────────────────────
@router.get("/status", response_model=StatusResponse)
def get_status():
    """返回 GPT-Neo / Router / TaskVectors 的加载状态"""
    from demo_system.algorithm.chapter4.adapter import _cache
    if not _cache['loaded']:
        return StatusResponse(loaded=False)
    return StatusResponse(
        loaded=True,
        base_model_params=sum(p.numel() for p in _cache['base_model'].parameters()),
        router_params=_cache['router'].count_trainable_params(),
        router_variant=_cache['router'].variant,
        task_vector_tasks=list(_cache['task_vectors'].keys()),
        head_tasks=list(_cache['heads'].keys()),
    )


# ────────────────────────────────────────
# GET /baseline — 4 任务基线准确率
# ────────────────────────────────────────
@router.get("/baseline", response_model=BaselineResponse)
def get_baseline():
    from demo_system.algorithm.chapter4.adapter import get_baseline as _get_bl
    return BaselineResponse(baseline=_get_bl())


# ────────────────────────────────────────
# GET /models — 模型详细信息列表
# ────────────────────────────────────────
@router.get("/models", response_model=list[ModelInfo])
def list_models():
    """返回第四章涉及的所有模型信息"""
    from demo_system.algorithm.chapter4.config import TASK_CONFIGS, BASELINE_ACC
    models = [
        ModelInfo(
            id="gpt-neo-125m", name="GPT-Neo 125M",
            type="pretrained", params="125.2M",
            description="EleutherAI GPT-Neo 125M 预训练语言模型 (Decoder-only Transformer)",
        ),
    ]
    # 微调模型 + TaskVector
    for task, cfg in TASK_CONFIGS.items():
        models.append(ModelInfo(
            id=f"ft-{task}", name=f"FT-{task.capitalize()}",
            type="finetuned", params="125.2M",
            description=cfg['description'],
            baseline_acc=BASELINE_ACC.get(task),
            num_classes=cfg['num_classes'],
        ))
        models.append(ModelInfo(
            id=f"tv-{task}", name=f"TaskVector-{task.capitalize()}",
            type="task_vector", params="~500MB",
            description=f"{task} 任务向量 (FT权重 − 基座权重, 160层)",
        ))
    # Router
    models.append(ModelInfo(
        id="router", name="AutoRouter",
        type="router", params="~99K trainable",
        description="双分支路由网络 — 分析输入分布，动态生成合并权重 α",
    ))
    # Merged
    models.append(ModelInfo(
        id="merged", name="Merged-Model",
        type="merged", params="125.2M",
        description="Router 自动合并模型: base + Σ(αᵢ × τᵢ)",
    ))
    return models


# ────────────────────────────────────────
# POST /upload-dataset — 上传数据集
# ────────────────────────────────────────
@router.post("/upload-dataset", response_model=UploadResponse)
async def upload_dataset(file: UploadFile = File(...)):
    """上传 CSV/JSON 数据集，返回解析预览"""
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ('.csv', '.json'):
        raise HTTPException(400, "仅支持 .csv 或 .json 格式")

    # 保存到临时路径 (文件名仅保留扩展名，用 uuid 前缀避免冲突)
    safe_name = f"{uuid.uuid4().hex[:8]}{ext}"
    save_path = _UPLOAD_DIR / safe_name

    try:
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(400, "文件大小不能超过 10MB")
        with open(save_path, "wb") as f:
            f.write(content)
    except HTTPException:
        raise
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"文件保存失败: {str(e)}")

    try:
        from demo_system.algorithm.chapter4.adapter import parse_uploaded_data
        from demo_system.algorithm.chapter4.config import TASK_NAMES
        records = parse_uploaded_data(str(save_path))

        # 截断到 1000 条
        if len(records) > 1000:
            records = records[:1000]

        task_dist = {}
        for name in TASK_NAMES:
            task_dist[name] = sum(1 for r in records if r['task_name'] == name)

        preview = records[:5]

        return UploadResponse(
            filename=safe_name,
            total_samples=len(records),
            task_distribution=task_dist,
            preview=preview,
        )
    except HTTPException:
        raise
    except Exception as e:
        # 清理失败的上传文件
        save_path.unlink(missing_ok=True)
        traceback.print_exc()
        raise HTTPException(400, f"数据集解析失败: {str(e)}")


# ────────────────────────────────────────
# GET /sample-dataset — 下载示例数据集
# ────────────────────────────────────────
@router.get("/sample-dataset")
def download_sample_dataset():
    """下载预置示例数据集 mixed_sample_50.csv"""
    if not _SAMPLE_DATASET.exists():
        raise HTTPException(404, "示例数据集不存在")
    return FileResponse(
        str(_SAMPLE_DATASET),
        filename="mixed_sample_50.csv",
        media_type="text/csv",
    )


# ────────────────────────────────────────
# POST /evaluate — 启动评测
# ────────────────────────────────────────
@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate(req: EvaluateRequest):
    """执行完整评测流程: 加载资源 → Router 推理 → 合并 → 评测"""
    file_path = _UPLOAD_DIR / req.file_path
    if not file_path.exists():
        raise HTTPException(404, f"数据集文件不存在: {req.file_path}")

    try:
        from demo_system.algorithm.chapter4.adapter import (
            load_resources, parse_uploaded_data, infer_and_evaluate_per_group,
        )

        # 加载资源
        load_resources(device='cuda')

        # 解析数据
        records = parse_uploaded_data(str(file_path))
        if len(records) > 1000:
            records = records[:1000]

        # 分组评测: router 分类 → 每组独立计算 α → 每组独立合并+评测
        result = infer_and_evaluate_per_group(records)

        return EvaluateResponse(
            alphas=result['alphas'],
            task_distribution=result['task_distribution'],
            task_classification_acc=round(result['task_classification_acc'], 4),
            per_task_recall=result['per_task_recall'],
            per_task_acc=result['per_task_acc'],
            per_task_norm_acc=result['per_task_norm_acc'],
            per_task_samples=result['per_task_samples'],
            overall_acc=result['overall_acc'],
            overall_norm_acc=result['overall_norm_acc'],
            baseline=result['baseline'],
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"评测失败: {str(e)}")


# ────────────────────────────────────────
# GET /graph — 知识图谱数据
# ────────────────────────────────────────
@router.get("/graph", response_model=GraphResponse)
def get_graph():
    """返回第四章知识图谱节点和边数据"""
    from demo_system.algorithm.chapter4.config import BASELINE_ACC

    nodes = [
        GraphNode(id="gpt-neo", name="GPT-Neo 125M", type="pretrained",
                  meta={"params": "125.2M", "arch": "GPTNeoForCausalLM (Decoder-only)",
                        "vocab": "50,257 tokens (BPE)", "hidden": "768 维 × 12 层 × 12 头",
                        "pretrain_data": "The Pile (825 GB 多领域语料)",
                        "pretrain_task": "Causal LM (自回归语言建模)",
                        "source": "EleutherAI/gpt-neo-125M",
                        "desc": "EleutherAI 开源的 GPT-Neo 125M 自回归语言模型，基于 Pile 数据集预训练"}),
        GraphNode(id="ft-code", name="FT-Code", type="finetuned",
                  meta={"task": "代码语言分类", "classes": 1006, "params": "125.2M",
                        "acc": f"{BASELINE_ACC['code']*100:.1f}%",
                        "train_epochs": "5 epochs", "lr": "2e-5", "batch_size": "32",
                        "desc": "判断代码片段所属的编程语言，覆盖 1006 种语言/框架"}),
        GraphNode(id="ft-langid", name="FT-LangID", type="finetuned",
                  meta={"task": "欧洲语言识别", "classes": 6, "params": "125.2M",
                        "acc": f"{BASELINE_ACC['langid']*100:.1f}%",
                        "train_epochs": "5 epochs", "lr": "2e-5", "batch_size": "32",
                        "desc": "识别文本所属的欧洲语言，涵盖英/法/德/西/意/葡 6 种"}),
        GraphNode(id="ft-law", name="FT-Law", type="finetuned",
                  meta={"task": "法律分类 (SCOTUS)", "classes": 13, "params": "125.2M",
                        "acc": f"{BASELINE_ACC['law']*100:.1f}%",
                        "train_epochs": "5 epochs", "lr": "2e-5", "batch_size": "32",
                        "desc": "对美国最高法院判决进行议题分类，覆盖 13 个法律领域"}),
        GraphNode(id="ft-math", name="FT-Math", type="finetuned",
                  meta={"task": "数学 QA 分类", "classes": 25, "params": "125.2M",
                        "acc": f"{BASELINE_ACC['math']*100:.1f}%",
                        "train_epochs": "5 epochs", "lr": "2e-5", "batch_size": "32",
                        "desc": "数学 QA 多项选择分类，涵盖 25 种数学概念类别"}),
        GraphNode(id="tv-code", name="τ-Code", type="task_vector",
                  meta={"desc": "代码任务向量 (FT-Code − GPT-Neo)", "layers": 160,
                        "formula": "τ = θ_ft − θ_base", "size": "~500 MB",
                        "source_model": "FT-Code"}),
        GraphNode(id="tv-langid", name="τ-LangID", type="task_vector",
                  meta={"desc": "语言识别任务向量 (FT-LangID − GPT-Neo)", "layers": 160,
                        "formula": "τ = θ_ft − θ_base", "size": "~500 MB",
                        "source_model": "FT-LangID"}),
        GraphNode(id="tv-law", name="τ-Law", type="task_vector",
                  meta={"desc": "法律任务向量 (FT-Law − GPT-Neo)", "layers": 160,
                        "formula": "τ = θ_ft − θ_base", "size": "~500 MB",
                        "source_model": "FT-Law"}),
        GraphNode(id="tv-math", name="τ-Math", type="task_vector",
                  meta={"desc": "数学任务向量 (FT-Math − GPT-Neo)", "layers": 160,
                        "formula": "τ = θ_ft − θ_base", "size": "~500 MB",
                        "source_model": "FT-Math"}),
        GraphNode(id="router", name="AutoRouter", type="router",
                  meta={"params": "~99K (可训练)", "variant": "hybrid_dual_branch",
                        "arch": "Embedding Branch + Task Branch → Fusion Head",
                        "output": "4 维 α 系数 (Softmax 归一化)",
                        "train_epochs": "20 epochs", "lr": "1e-3",
                        "desc": "双分支路由网络：Embedding 分支编码输入特征，Task 分支分析任务分布，融合后输出合并权重 α"}),
        GraphNode(id="merged", name="Merged-Model", type="merged",
                  meta={"params": "125.2M", "formula": "θ = θ_base + Σ(αᵢ × τᵢ)",
                        "source_models": "4 个任务向量 (code/langid/law/math)",
                        "routing": "AutoRouter 动态权重",
                        "desc": "Router 根据输入数据分布动态生成 α，按加权求和合并任务向量到基座模型"}),
    ]
    edges = [
        # 微调
        GraphEdge(source="gpt-neo", target="ft-code", relation="微调", style="solid"),
        GraphEdge(source="gpt-neo", target="ft-langid", relation="微调", style="solid"),
        GraphEdge(source="gpt-neo", target="ft-law", relation="微调", style="solid"),
        GraphEdge(source="gpt-neo", target="ft-math", relation="微调", style="solid"),
        # 模块化 (task vector)
        GraphEdge(source="ft-code", target="tv-code", relation="任务向量", style="dashed"),
        GraphEdge(source="ft-langid", target="tv-langid", relation="任务向量", style="dashed"),
        GraphEdge(source="ft-law", target="tv-law", relation="任务向量", style="dashed"),
        GraphEdge(source="ft-math", target="tv-math", relation="任务向量", style="dashed"),
        # Router 路由
        GraphEdge(source="tv-code", target="router", relation="路由", style="dotted"),
        GraphEdge(source="tv-langid", target="router", relation="路由", style="dotted"),
        GraphEdge(source="tv-law", target="router", relation="路由", style="dotted"),
        GraphEdge(source="tv-math", target="router", relation="路由", style="dotted"),
        # 合并
        GraphEdge(source="router", target="merged", relation="合并", style="solid"),
        GraphEdge(source="gpt-neo", target="merged", relation="基座", style="solid"),
    ]
    return GraphResponse(nodes=nodes, edges=edges)
