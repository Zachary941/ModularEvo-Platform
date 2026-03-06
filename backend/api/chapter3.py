"""第三章 API 路由

提供模块管理、模型加载、合并评测、知识图谱数据接口。
"""
import sys
import json
import traceback
from pathlib import Path
from fastapi import APIRouter, BackgroundTasks, HTTPException

# 确保 demo_system 父目录在 sys.path 中，支持 algorithm 包导入
_demo_root = Path(__file__).resolve().parent.parent.parent
_workspace_root = _demo_root.parent
if str(_workspace_root) not in sys.path:
    sys.path.insert(0, str(_workspace_root))

from .ch3_schemas import (
    ModuleInfo, LoadModuleRequest, LoadModuleResponse,
    FinetunedInfo, EvalResult,
    MergeRequest, MergeResponse,
    GraphResponse, GraphNode, GraphEdge,
)

router = APIRouter()

# ── 缓存已加载的资源 ──
_cache = {
    'base_model': None, 'config': None, 'tokenizer': None,
    'modules': {},       # language -> LoadModuleResponse
    'finetuned': {},     # task -> model
    'eval_results': {},  # task -> metrics dict
    'merge_task': None,  # 当前合并任务状态
}


def _ensure_base():
    """确保 CodeBERT 基座已加载（延迟加载）"""
    if _cache['base_model'] is None:
        from demo_system.algorithm.chapter3.model_loader import load_base_model
        model, config, tokenizer = load_base_model('cuda')
        _cache['base_model'] = model
        _cache['config'] = config
        _cache['tokenizer'] = tokenizer
    return _cache['base_model'], _cache['config'], _cache['tokenizer']


# ────────────────────────────────────────
# P2.1  GET /modules — 可用模块列表
# ────────────────────────────────────────
@router.get("/modules", response_model=list[ModuleInfo])
def list_modules():
    """返回可用的语言模块列表"""
    from demo_system.algorithm.chapter3.model_loader import get_module_info
    return [get_module_info(lang) for lang in ('java', 'python')]


# ────────────────────────────────────────
# P2.2  POST /load-module — 加载模块
# ────────────────────────────────────────
@router.post("/load-module", response_model=LoadModuleResponse)
def load_module(req: LoadModuleRequest):
    """加载指定语言的模块化 mask，返回稀疏率统计"""
    if req.language not in ('java', 'python'):
        raise HTTPException(400, f"不支持的语言: {req.language}")

    # 如果已缓存直接返回
    if req.language in _cache['modules']:
        return _cache['modules'][req.language]

    try:
        from demo_system.algorithm.chapter3.model_loader import load_sparse_module
        base, _, _ = _ensure_base()
        result = load_sparse_module(req.language, base, 'cuda')
        resp = LoadModuleResponse(
            language=req.language,
            sparsity=result['sparsity'],
            wrr=result['wrr'],
            layer_stats=result['layer_stats'],
        )
        _cache['modules'][req.language] = resp
        return resp
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"模块加载失败: {str(e)}")


# ────────────────────────────────────────
#   GET /finetuned — 微调模型信息 + 评测
# ────────────────────────────────────────
@router.get("/finetuned", response_model=list[FinetunedInfo])
def list_finetuned():
    """返回可用微调模型信息"""
    from demo_system.algorithm.chapter3.config import FINETUNED_PATHS
    import os
    infos = []
    for task, path in FINETUNED_PATHS.items():
        infos.append(FinetunedInfo(
            task=task,
            params_m=125.8 if task == 'clone_detection' else 127.0,
            checkpoint=os.path.basename(path),
        ))
    return infos


@router.post("/evaluate/{task}", response_model=EvalResult)
def evaluate_finetuned(task: str):
    """对指定任务的微调模型执行评测"""
    if task not in ('clone_detection', 'code_search'):
        raise HTTPException(400, f"不支持的任务: {task}")

    # 缓存评测结果
    if task in _cache['eval_results']:
        return EvalResult(task=task, metrics=_cache['eval_results'][task])

    try:
        from demo_system.algorithm.chapter3.model_loader import load_finetuned_model
        from demo_system.algorithm.chapter3.evaluator import evaluate_task
        base, config, tokenizer = _ensure_base()
        model = load_finetuned_model(task, base, config, tokenizer, 'cuda')
        metrics = evaluate_task(task, model, tokenizer)
        # 确保所有值都是 float
        metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        _cache['eval_results'][task] = metrics
        return EvalResult(task=task, metrics=metrics)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"评测失败: {str(e)}")


# ────────────────────────────────────────
# P2.3  POST /merge — 合并 + 评测
# ────────────────────────────────────────
@router.post("/merge", response_model=MergeResponse)
def merge_and_eval(req: MergeRequest):
    """执行模型合并，返回各任务的评测结果"""
    from demo_system.algorithm.chapter3.config import MERGE_METHODS
    if req.method not in MERGE_METHODS:
        raise HTTPException(400, f"不支持的合并方法: {req.method}. 可选: {list(MERGE_METHODS.keys())}")

    try:
        from demo_system.algorithm.chapter3.merger import merge_and_evaluate
        merge_kwargs = {}
        if req.method == 'ties':
            merge_kwargs['param_value_mask_rate'] = req.param_value_mask_rate
        elif req.method == 'dare':
            merge_kwargs['weight_mask_rates'] = req.weight_mask_rates

        result = merge_and_evaluate(
            req.method,
            scaling_coefficients=req.scaling_coefficients,
            device='cuda',
            **merge_kwargs,
        )
        # 确保 metrics 值为 float
        cleaned_results = {}
        for task_name, metrics in result['results'].items():
            cleaned_results[task_name] = {k: round(float(v), 4) for k, v in metrics.items()}

        return MergeResponse(
            method=result['method'],
            scaling_coefficients=result['scaling_coefficients'],
            results=cleaned_results,
        )
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(500, f"合并失败: {str(e)}")


# ────────────────────────────────────────
# P2.4  GET /graph — 知识图谱数据
# ────────────────────────────────────────
@router.get("/graph", response_model=GraphResponse)
def get_graph():
    """返回第三章知识图谱节点和边数据"""
    nodes = [
        GraphNode(id="codebert", name="CodeBERT", type="pretrained",
                  meta={"params": "124.6M", "desc": "预训练代码语言模型"}),
        GraphNode(id="module-java", name="Module-Java", type="module",
                  meta={"wrr": "22.94%", "sparsity": "77.06%", "desc": "Java 语言模块"}),
        GraphNode(id="module-python", name="Module-Python", type="module",
                  meta={"wrr": "24.15%", "sparsity": "75.85%", "desc": "Python 语言模块"}),
        GraphNode(id="ft-clone", name="FT-CloneDet", type="finetuned",
                  meta={"task": "克隆检测", "params": "125.8M"}),
        GraphNode(id="ft-search", name="FT-CodeSearch", type="finetuned",
                  meta={"task": "代码搜索", "params": "127.0M"}),
        GraphNode(id="merged", name="Merged-Model", type="merged",
                  meta={"desc": "合并后模型"}),
    ]
    edges = [
        GraphEdge(source="codebert", target="module-java", relation="模块化", style="dashed"),
        GraphEdge(source="codebert", target="module-python", relation="模块化", style="dashed"),
        GraphEdge(source="module-java", target="ft-clone", relation="模块微调", style="dotted"),
        GraphEdge(source="module-python", target="ft-search", relation="模块微调", style="dotted"),
        GraphEdge(source="ft-clone", target="merged", relation="合并", style="solid"),
        GraphEdge(source="ft-search", target="merged", relation="合并", style="solid"),
    ]
    return GraphResponse(nodes=nodes, edges=edges)
