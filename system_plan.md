# TransModular Demo 系统开发手册

> **版本**: v0.3 (技术细节确认后更新)  
> **更新日期**: 2026-03-06  
> **技术栈**: FastAPI + Vue 3 + Element Plus + ECharts + SQLite  

---

## 一、系统概述

### 1.1 项目目标

构建一个用于**展示 Transformer 模型模块化与动态合并**算法的 Demo 系统，涵盖论文第三章（基于 CodeBERT 的模块化方法）与第四章（基于 GPT-Neo 的 Router 动态合并方法）的核心流程，以可视化、可交互的方式呈现给用户。

### 1.2 核心展示内容

| 页面 | 章节 | 基座模型 | 展示流程 |
|------|------|----------|----------|
| 页面 A：模块化全流程 | 第三章 | CodeBERT (codebert-base) | 模块化 → 下游任务微调 → 模型合并 |
| 页面 B：Router 动态合并 | 第四章 | GPT-Neo 125M | 预部署模型 + Router → 用户上传数据集 → 评测输出 |

### 1.3 页面通用元素

- **顶部知识图谱**：每个页面顶部展示一个交互式知识图谱，用节点表示模型/模块，用有向边表示关系（微调、模块化、模块微调、合并）。
- **无需用户认证**：内部展示系统，不设置登录功能。

---

## 二、系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────┐
│                  前端 (Vue 3)                │
│  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
│  │ 知识图谱  │  │ 页面A    │  │ 页面B     │  │
│  │ (D3.js/  │  │ 模块化   │  │ Router    │  │
│  │  ECharts)│  │ 全流程   │  │ 动态合并  │  │
│  └──────────┘  └──────────┘  └───────────┘  │
└──────────────────┬──────────────────────────┘
                   │ REST API / WebSocket
┌──────────────────▼──────────────────────────┐
│               后端 (FastAPI)                 │
│  ┌──────────────────────────────────────┐   │
│  │ API 层: 任务管理 / 模型管理 / 评测   │   │
│  ├──────────────────────────────────────┤   │
│  │ 服务层: 模块化引擎 / 合并引擎 /     │   │
│  │         Router推理 / 评测服务        │   │
│  ├──────────────────────────────────────┤   │
│  │ 算法层: Tran_SeaM / TransModular_GPT│   │
│  └──────────────────────────────────────┘   │
│  ┌──────────┐  ┌───────────────────────┐    │
│  │ SQLite   │  │ 文件存储 (模型/数据)  │    │
│  └──────────┘  └───────────────────────┘    │
└─────────────────────────────────────────────┘
```

### 2.2 技术选型

| 层次 | 技术 | 说明 |
|------|------|------|
| 前端框架 | Vue 3 + Vite | 轻量 SPA 框架 |
| UI 组件库 | Element Plus | 简洁美观，中文文档完善 |
| 知识图谱可视化 | ECharts (关系图) | 中文文档完善、Vue 集成方便、交互丰富 |
| 后端框架 | FastAPI | 异步、自动生成 OpenAPI 文档 |
| 数据库 | SQLite | 轻量，存储任务记录、评测结果 |
| 任务队列 | FastAPI BackgroundTasks | Demo 阶段使用轻量方案，预留 Celery 扩展 |
| 深度学习框架 | PyTorch + Transformers | 复用现有算法代码 |

### 2.3 目录结构规划

```
demo_system/
├── backend/                     # FastAPI 后端
│   ├── main.py                  # 应用入口
│   ├── api/                     # API 路由
│   │   ├── chapter3.py          # 第三章相关接口
│   │   ├── chapter4.py          # 第四章相关接口
│   │   └── graph.py             # 知识图谱数据接口
│   ├── services/                # 业务逻辑层
│   │   ├── modularizer_svc.py   # 模块化服务 (封装 Tran_SeaM)
│   │   ├── merge_svc.py         # 模型合并服务
│   │   ├── router_svc.py        # Router 推理服务 (封装 TransModular_GPT)
│   │   └── eval_svc.py          # 评测服务
│   ├── models/                  # 数据库模型 (SQLAlchemy)
│   │   └── schemas.py           # Pydantic 请求/响应模型
│   ├── core/                    # 配置、依赖注入
│   │   └── config.py
│   └── tasks/                   # 后台任务
│       └── workers.py
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── views/
│   │   │   ├── Chapter3View.vue # 第三章页面
│   │   │   ├── Chapter4View.vue # 第四章页面
│   │   │   └── HomeView.vue     # 首页
│   │   ├── components/
│   │   │   ├── KnowledgeGraph.vue  # 知识图谱组件
│   │   │   ├── ModuleFlowChart.vue # 流程可视化
│   │   │   └── EvalResult.vue      # 评测结果展示
│   │   └── api/                 # Axios 接口封装
│   └── ...
├── algorithm/                   # 算法适配层 (自包含，不依赖外部代码)
│   ├── __init__.py
│   ├── chapter3/                # 第三章适配（多文件，按职责拆分）
│   │   ├── __init__.py
│   │   ├── config.py            # 路径配置 & 预置资源路径常量
│   │   ├── model_loader.py      # 模型加载：CodeBERT / mask模块 / 微调模型
│   │   ├── modularizer.py       # 模块化训练逻辑 (预留接口)
│   │   ├── evaluator.py         # 评测逻辑
│   │   ├── merger.py            # 合并逻辑
│   │   └── libs/                # 从原项目复制的核心依赖代码
│   │       ├── __init__.py
│   │       ├── mask_layer.py        # ← Tran_SeaM/mask_layer.py
│   │       ├── sparse_utils.py      # ← Tran_SeaM/utils.py (摘取 load_init_module_sparse)
│   │       ├── clone_model.py       # ← task_eval/code_clone_eval.py (Model + evaluate)
│   │       ├── search_model.py      # ← task_eval/nl_code_search_eval.py (Model + evaluate)
│   │       ├── merging_methods.py   # ← merge_methods/merging_methods.py (MergingMethod)
│   │       ├── task_vector.py       # ← merge_methods/task_vector.py (TaskVector)
│   │       ├── mask_weights_utils.py # ← merge_methods/mask_weights_utils.py
│   │       └── merge_utils.py       # ← merge_utils/merge_utils.py (公共工具)
│   └── chapter4/                # 第四章适配
│       ├── __init__.py
│       ├── adapter.py           # 适配层封装
│       └── libs/                # 从 TransModular_GPT/router 复制的核心代码
│           ├── __init__.py
│           ├── config.py            # ← router/config.py (路径已改为本地)
│           ├── router.py            # ← router/router.py (Router 网络)
│           ├── merge.py             # ← router/merge.py (动态合并)
│           ├── task_vectors.py      # ← router/task_vectors.py (任务向量加载)
│           └── data.py              # ← router/data.py (数据预处理)
├── data/                        # 所有模型和数据集 (自包含)
│   ├── models/                  # 模型文件 (每个模型一个子目录)
│   │   ├── codebert-base/           # CodeBERT 预训练基座
│   │   ├── module_java/             # Java 模块化 mask (result/ 子目录)
│   │   ├── module_python/           # Python 模块化 mask (result/ 子目录)
│   │   ├── finetuned_clone/         # 克隆检测微调模型 checkpoint
│   │   ├── finetuned_search/        # 代码搜索微调模型 checkpoint
│   │   ├── gpt-neo-125m/            # GPT-Neo 125M 预训练基座
│   │   ├── task_vectors/            # 第四章预计算任务向量 (4个 .pt)
│   │   ├── heads/                   # 第四章分类头 (4个 .pt)
│   │   └── router_checkpoint/       # Router 训练好的 checkpoint
│   └── datasets/                # 评测数据集
│       ├── clone_detection/         # test.txt + data.jsonl
│       ├── code_search/             # cosqa_dev.json
│       └── sample_datasets/         # mixed_sample_50.csv (第四章示例)
└── docker-compose.yml           # 容器化部署 (可选)
```

### 2.4 首页 (HomeView) 设计

首页展示系统简介 + 两个章节入口卡片：

```
┌───────────────────────────────────────────────────┐
│ TransModular Demo                                 │
│ Transformer 模型模块化与动态合并展示系统           │
│                                                   │
│ [系统简介: 1-2 句话说明系统用途]               │
│                                                   │
│ ┌─────────────────────┐ ┌─────────────────────┐ │
│ │  第三章               │ │  第四章               │ │
│ │  CodeBERT 模块化      │ │  GPT-Neo Router       │ │
│ │  全流程展示           │ │  动态合并展示         │ │
│ │                     │ │                     │ │
│ │  模块化 → 微调 → 合并│ │  上传数据 → 评测    │ │
│ │  [进入 ▶]            │ │  [进入 ▶]            │ │
│ └─────────────────────┘ └─────────────────────┘ │
└───────────────────────────────────────────────────┘
```

- 简介文字: 简洁说明系统的目的（展示 Transformer 模型模块化与合并算法）
- 卡片 A: 第三章入口，显示“CodeBERT 模块化全流程”，点击跳转到 `/chapter3`
- 卡片 B: 第四章入口，显示“GPT-Neo Router 动态合并”，点击跳转到 `/chapter4`

---

## 三、页面 A：第三章 — CodeBERT 模块化全流程展示

### 3.1 页面布局

```
┌─────────────────────────────────────────────────────────┐
│ [知识图谱区域]                                         │
│ CodeBERT ──模块化──▶ Module-Java   Module-Python       │
│                         │               │              │
│                      模块微调        模块微调           │
│                         ▼               ▼              │
│                   FT-CloneDet(Java) FT-Search(Python)  │
│                         ╲              ╱               │
│                          ╲    合并    ╱                │
│                           ▼         ▼                  │
│                         Merged-Model                   │
└─────────────────────────────────────────────────────────┘
┌─────────────────────────────────────────────────────────┐
│ Step 1: 模块化        │ Step 2: 微调     │ Step 3: 合并 │
│ [快速演示模式]        │ [任务配置面板]   │ [合并配置]   │
│ 加载预训练 mask:      │ ☑ 克隆检测(Java) │ - 合并方法   │
│  ☑ Java 模块          │ ☑ 代码搜索(Py)   │ - 缩放系数   │
│  ☑ Python 模块        │ [加载预微调模型] │ [开始合并]   │
│ [加载模块]            │                  │              │
│                       │ [结果展示]       │ [结果展示]   │
│ [模块稀疏率/热力图]   │  F1 / Precision  │ 准确率对比   │
└─────────────────────────────────────────────────────────┘
```

### 3.2 功能模块

#### 3.2.1 Step 1 — 模块化 (Modularization)

- **默认模式**: 快速演示模式 — 直接加载预训练好的 mask 文件
- **可选语言模块**: Java（用于克隆检测）、Python（用于代码搜索）
- **后端**: 调用 `Tran_SeaM/utils.py` 的 `load_init_module_sparse()` 加载预训练 mask
- **预留接口**: `modularize()` 接口保留完整训练逻辑（调用 `modularizer.py`），供后续扩展
- **输出展示**:
  - 权重保留率 (Weight Retention Rate) 逐层可视化
  - 最终稀疏掩码热力图
  - 模块参数量 & 稀疏率统计

#### 3.2.2 Step 2 — 下游任务微调

- **固定任务** (两个任务均需展示，合并阶段需要至少 2 个任务模型):
  - 代码克隆检测 (Clone Detection, BigCloneBench) — 使用 Java 模块
  - 自然语言代码搜索 (NL Code Search) — 使用 Python 模块
- **默认模式**: 加载预微调好的模型文件，直接展示评测结果
- **预留接口**: `finetune()` 接口保留完整微调逻辑，供后续扩展
- **后端**: 调用 `Tran_SeaM/task_merge/task_eval/` 相关代码进行评测
- **输出展示**:
  - 评测指标: 克隆检测 → F1, 代码搜索 → Precision
  - 模型详情: 参数量、训练配置

#### 3.2.3 Step 3 — 模型合并

- **合并方法选择** (4 种): 
  - Task Arithmetic
  - TIES Merging
  - DARE (对应代码中的 `mask_merging`)
  - ModularEvo (本文方法，需新增实现)
- **输入**: 选择合并方法、缩放系数
- **合并对象**: 步骤 2 中的两个微调模型 (克隆检测-Java + 代码搜索-Python)
- **后端**: 调用 `Tran_SeaM/task_merge/merge_lm.py`
- **输出展示**:
  - 合并后模型在两个任务上的准确率对比表 (F1 + Precision)
  - 4 种合并方法的性能对比图
  - 模型参数量变化

### 3.3 知识图谱（页面 A）

- **节点**:
  - CodeBERT (预训练模型)
  - Module-Java (Java 语言模块)
  - Module-Python (Python 语言模块)
  - FT-CloneDet (克隆检测微调模型)
  - FT-CodeSearch (代码搜索微调模型)
  - Merged-Model (合并后模型)
- **边类型** (4 种):
  - `模块化` (CodeBERT → Module-Java / Module-Python)
  - `模块微调` (Module-Java → FT-CloneDet, Module-Python → FT-CodeSearch)
  - `合并` (FT-CloneDet + FT-CodeSearch → Merged-Model)
- **交互**: 点击节点查看模型详情（参数量、稀疏率、性能指标）

### 3.4 后端 API 设计

```
POST   /api/ch3/modularize          # 启动模块化任务
GET    /api/ch3/modularize/{task_id} # 查询模块化进度
POST   /api/ch3/finetune            # 启动微调任务
GET    /api/ch3/finetune/{task_id}   # 查询微调进度
POST   /api/ch3/merge               # 执行模型合并
GET    /api/ch3/merge/{task_id}      # 查询合并结果
GET    /api/ch3/models               # 获取可用模型列表
GET    /api/ch3/graph                # 获取知识图谱数据
```

---

## 四、页面 B：第四章 — GPT-Neo Router 动态合并展示

### 4.1 页面布局

```
┌──────────────────────────────────────────────────┐
│ [知识图谱区域]                                    │
│           GPT-Neo-125M (Base)                    │
│          ╱    ╱     ╲     ╲                      │
│       微调  微调   微调   微调                    │
│        ▼     ▼      ▼      ▼                     │
│     Code  LangID  Law   Math                     │
│        ╲     ╲      ╱     ╱                      │
│         ▼ Router 动态合并 ▼                       │
│           Merged Model                           │
└──────────────────────────────────────────────────┘
┌──────────────────────────────────────────────────┐
│ 左侧: 预部署模型信息            右侧: 评测区域   │
│ ┌──────────────────────┐ ┌─────────────────────┐ │
│ │ 基座模型: GPT-Neo    │ │ 上传自定义数据集    │ │
│ │ 任务模型:            │ │ [选择文件] [上传]   │ │
│ │  ✅ Code (1006类)    │ │                     │ │
│ │  ✅ LangID (6类)     │ │ 数据集格式要求:     │ │
│ │  ✅ Law (13类)       │ │ CSV/JSON, 含 text   │ │
│ │  ✅ Math (25类)      │ │ 和 label 字段       │ │
│ │ Router: ✅ 已加载    │ │                     │ │
│ │                      │ │ [开始评测]          │ │
│ │ 任务向量 α 系数:     │ │                     │ │
│ │ code=0.49 langid=... │ │ [评测结果区域]      │ │
│ └──────────────────────┘ │ - 任务分类准确率    │ │
│                          │ - 每任务 α 系数     │ │
│ [模型性能基线展示]       │ - 混淆矩阵          │ │
│ Code: 83.90%             │ - 性能对比图        │ │
│ LangID: 91.73%           │                     │ │
│ Law: 70.57%              │                     │ │
│ Math: 95.85%             │                     │ │
└──────────────────────────┴─────────────────────┘ │
└──────────────────────────────────────────────────┘
```

### 4.2 功能模块

#### 4.2.1 预部署资源展示

- **展示内容**:
  - 基座模型信息 (GPT-Neo 125M, 参数量)
  - 4 个微调后模型状态 (code/langid/law/math)
  - Router 网络参数信息 (~98,952 参数)
  - 各任务基线准确率
- **数据来源**: 预加载的配置与评测结果

#### 4.2.2 自定义数据集上传与评测

- **上传**: 用户上传 CSV/JSON 格式数据集（4 个已知任务的混合数据）
  - 必需字段: `text` (输入文本), `label` (分类标签), `task_id` (任务标识: 0=code/1=langid/2=law/3=math)
  - 后端限制: 最大 200 条样本（超出自动截断，UI 中不显性提示）
- **评测流程**:
  1. 数据预处理 & Tokenization
  2. Router 网络推理 → 计算任务分布 α（数据集级别）
  3. 按 α 动态合并任务向量 → 生成合并模型
  4. 合并模型推理 → 分类结果
  5. 对比基线模型 → 输出评测报告
- **输出** (聚合指标为主，后续可扩展逐样本展示):
  - 任务识别准确率 & 各任务 Recall
  - 各任务分类准确率
  - Router 学习到的 α 系数可视化
  - 与单任务基线的对比柱状图

#### 4.2.3 可视化组件

- α 系数柱状图
- 性能对比表格 (Merged vs. Baseline per task)
- 任务识别准确率 & Recall 汇总表

### 4.3 知识图谱（页面 B）

- **节点**:
  - GPT-Neo 125M (预训练基座)
  - FT-Code / FT-LangID / FT-Law / FT-Math (微调模型)
  - TaskVector-{task} (任务向量)
  - Router (路由网络)
  - Merged-Model (动态合并结果)
- **边类型** (4 种):
  - `微调` (GPT-Neo → FT-Model)
  - `模块化` (FT-Model → TaskVector, 即 τ = θ_ft - θ_base)
  - `合并` (TaskVector × N + Router → Merged-Model)
- **交互**: 
  - 点击 TaskVector 节点显示其稀疏率、层数信息
  - 点击 Router 节点显示当前 α 系数分布
  - 合并结果节点随评测结果动态更新

### 4.4 后端 API 设计

```
GET    /api/ch4/status               # 获取预部署模型/Router状态
GET    /api/ch4/baseline             # 获取基线评测结果
POST   /api/ch4/upload-dataset       # 上传自定义数据集
POST   /api/ch4/evaluate             # 启动评测任务
GET    /api/ch4/evaluate/{task_id}   # 查询评测进度与结果
GET    /api/ch4/graph                # 获取知识图谱数据
```

---

## 五、知识图谱组件设计

### 5.1 数据模型

```json
{
  "nodes": [
    {"id": "codebert", "name": "CodeBERT", "type": "pretrained", "params": "125M", "meta": {}},
    {"id": "module-python", "name": "Module-Python", "type": "module", "sparsity": "75.82%", "meta": {}},
    {"id": "ft-clone", "name": "FT-CloneDet", "type": "finetuned", "accuracy": "95.2%", "meta": {}}
  ],
  "edges": [
    {"source": "codebert", "target": "module-python", "relation": "模块化", "style": "dashed"},
    {"source": "codebert", "target": "ft-clone", "relation": "微调", "style": "solid"},
    {"source": "module-python", "target": "ft-clone-module", "relation": "模块微调", "style": "dotted"}
  ]
}
```

### 5.2 可视化方案

- **方案**: ECharts 关系图 (`graph` 类型)
  - 优势: 中文文档完善、与 Vue 集成方便 (vue-echarts)、交互丰富
  - 力导向布局 (Force Layout)
  - 4 种关系类型使用不同线型和颜色:
    - `微调`: 实线 (蓝色)
    - `模块化`: 虚线 (绿色)
    - `模块微调`: 点线 (橙色)
    - `合并`: 粗实线 (红色)

### 5.3 交互行为

- 鼠标悬停节点: 显示 Tooltip (参数量、稀疏率、准确率等)
- 点击节点: 展开详情侧边栏
- 节点动态状态: 训练中(动画)、已完成(高亮)、待处理(灰色)
- 实时更新: 任务完成后自动添加新节点/边

---

## 六、数据库设计

### 6.1 核心表

```sql
-- 任务记录
CREATE TABLE tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chapter     INTEGER NOT NULL,          -- 3 or 4
    task_type   TEXT NOT NULL,             -- modularize / finetune / merge / evaluate
    status      TEXT DEFAULT 'pending',    -- pending / running / completed / failed
    params      TEXT,                      -- JSON: 任务参数
    result      TEXT,                      -- JSON: 任务结果
    log_path    TEXT,                      -- 日志文件路径
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 模型注册
CREATE TABLE models (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL,
    model_type  TEXT NOT NULL,             -- pretrained / module / finetuned / merged
    chapter     INTEGER NOT NULL,
    path        TEXT NOT NULL,             -- 模型文件路径
    meta        TEXT,                      -- JSON: 参数量、稀疏率、来源等
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 评测结果
CREATE TABLE evaluations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     INTEGER REFERENCES tasks(id),
    model_id    INTEGER REFERENCES models(id),
    dataset     TEXT,
    metrics     TEXT,                      -- JSON: {accuracy, f1, mrr, ...}
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

---

## 七、算法适配层设计

### 7.1 第三章适配 (algorithm/chapter3/)

将 `Tran_SeaM/` 下的核心代码封装为 **5 个模块文件**，各司其职，互相独立调用：

#### 7.1.0 文件职责总览

```
algorithm/chapter3/
├── config.py          # 路径常量 + 设备配置
├── model_loader.py    # 模型加载统一入口
├── modularizer.py     # 模块化训练逻辑
├── evaluator.py       # 评测逻辑（克隆检测 + 代码搜索）
└── merger.py          # 模型合并逻辑（4 种方法）
```

#### 7.1.1 config.py — 路径配置

> 所有路径均指向 `demo_system/data/` 下的本地副本，**不依赖外部文件夹**。

```python
import os

# demo_system/ 根目录
DEMO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
DATA_ROOT = os.path.join(DEMO_ROOT, "data")
MODEL_ROOT = os.path.join(DATA_ROOT, "models")
DATASET_ROOT = os.path.join(DATA_ROOT, "datasets")

# ── 预训练基座 ──
CODEBERT_PATH = os.path.join(MODEL_ROOT, "codebert-base")

# ── 模块化 mask 路径 ──
MODULE_PATHS = {
    "java":   os.path.join(MODEL_ROOT, "module_java"),
    "python": os.path.join(MODEL_ROOT, "module_python"),
}

# ── 微调模型 checkpoint 路径 ──
FINETUNED_PATHS = {
    "clone_detection": os.path.join(MODEL_ROOT, "finetuned_clone", "pytorch_model.bin"),
    "code_search":     os.path.join(MODEL_ROOT, "finetuned_search", "pytorch_model.bin"),
}

# ── 评测数据路径 ──
EVAL_DATA_PATHS = {
    "clone_detection": os.path.join(DATASET_ROOT, "clone_detection", "test.txt"),
    "code_search":     os.path.join(DATASET_ROOT, "code_search", "cosqa_dev.json"),
}

# ── 克隆检测数据附属文件 ──
CLONE_DATA_JSONL = os.path.join(DATASET_ROOT, "clone_detection", "data.jsonl")

# ── 合并方法名映射 (前端显示名 → MergingMethod 内部名) ──
MERGE_METHODS = {
    "task_arithmetic": "task_arithmetic",
    "ties":           "ties_merging",
    "dare":           "mask_merging",
    # "modular_evo":  需要新增实现
}
```

#### 7.1.2 model_loader.py — 模型加载

统一封装 CodeBERT 基座、Mask 模块、微调模型的加载逻辑，供其他模块调用。

```python
"""模型加载统一入口"""
# 依赖 (均已复制到 libs/ 目录下):
#   libs/sparse_utils.py → load_init_module_sparse()
#   libs/clone_model.py  → CloneModel + RobertaClassificationHead
#   libs/search_model.py → SearchModel + MLP head

def load_base_model(device="cuda") -> tuple[RobertaModel, RobertaConfig, RobertaTokenizer]:
    """加载 CodeBERT 基座模型"""

def load_sparse_module(language: str, base_model, device="cuda") -> dict:
    """加载预训练 mask 并应用到基座模型，返回 ModuleInfo
    调用: libs/sparse_utils.py → load_init_module_sparse()
    返回: {model, sparsity, wrr, layer_stats: [{name, total, nonzero, ratio}]}
    """

def load_finetuned_model(task: str, base_model, config, tokenizer, device="cuda") -> nn.Module:
    """加载指定任务的微调模型
    task: 'clone_detection' | 'code_search'
    根据 task 选择 CloneModel 或 SearchModel 架构，加载 checkpoint
    """

def get_module_info(language: str) -> dict:
    """获取模块元信息（不加载到 GPU），返回 {path, wrr, exists}"""
```

**关键实现细节**:
- `load_sparse_module()` 内部调用 `libs/sparse_utils.load_init_module_sparse(model, module_path, prefix='roberta.')`，遍历 state_dict 中的 `*_mask` 键，二值化后与权重相乘
- 逐层收集 `(total_params, nonzero_params, ratio)` 统计信息，用于前端热力图展示
- 克隆检测模型使用 `RobertaClassificationHead`（hidden_size×2 → 2），代码搜索模型使用 MLP Siamese 架构（768×4 → 768 → 1）

#### 7.1.3 modularizer.py — 模块化训练（预留）

```python
"""模块化训练逻辑 — P0 阶段为快速演示模式（仅加载），预留训练接口"""
# 参考原代码: Tran_SeaM/modularizer.py (Modularizer extends Trainer)
#       已复制到: libs/mask_layer.py (MaskLinear, init_mask_model, Binarization)

def modularize(language: str, lr=0.001, alpha=10.0, epochs=4) -> dict:
    """完整模块化训练（预留接口，当前直接抛 NotImplementedError）
    训练逻辑: 
      1. init_mask_model() 将 Linear → MaskLinear，冻结权重只训练 mask
      2. Modularizer(Trainer) 使用 loss = MLM_loss + alpha * WRR_loss 训练
      3. Binarization: mask > 0 → 1, 否则 → 0 (straight-through estimator)
      4. 输出: pytorch_model.bin (含 weight_mask / bias_mask)
    """
    raise NotImplementedError("模块化训练暂不支持，请使用 load_sparse_module() 加载预训练 mask")
```

#### 7.1.4 evaluator.py — 评测逻辑

```python
"""评测逻辑封装"""
# 依赖 (均已复制到 libs/ 目录下):
#   libs/clone_model.py  → evaluate() (克隆检测评测)
#   libs/search_model.py → evaluate() (代码搜索评测)

def evaluate_clone_detection(model, tokenizer, test_data_file=None) -> dict:
    """评测克隆检测任务
    内部调用: code_clone_eval.evaluate(model, tokenizer, test_data_file, output_dir)
    数据格式: test.txt (TSV: url1 \t url2 \t label) + data.jsonl ({idx, func})
    返回: {eval_f1, eval_precision, eval_recall, eval_threshold}
    """

def evaluate_code_search(model, tokenizer, eval_data_file=None) -> dict:
    """评测代码搜索任务
    内部调用: nl_code_search_eval.evaluate(model, tokenizer, eval_data_file, output_dir)
    数据格式: cosqa_dev.json ([{idx, query, doc, code, label}])
    返回: {acc, precision, recall, f1, acc_and_f1}
    """

def evaluate_model(model, task: str, tokenizer) -> dict:
    """统一评测入口，根据 task 分发到对应评测函数"""
```

**关键实现细节**:
- 克隆检测评测使用 `multiprocessing.Pool(16)` 并行加载数据，batch_size=4
- 克隆检测通过搜索 threshold (1/100 ~ 99/100) 找最优 F1 对应的阈值
- 代码搜索评测 batch_size=8，使用 BCELoss + 0.5 阈值二分类
- 两个评测函数签名一致: `evaluate(model, tokenizer, data_file, output_dir) → dict`

#### 7.1.5 merger.py — 模型合并

```python
"""模型合并逻辑封装"""
# 依赖 (均已复制到 libs/ 目录下):
#   libs/merging_methods.py → MergingMethod 类
#   libs/task_vector.py     → TaskVector 类

def merge_models(
    method: str,                     # 'task_arithmetic' | 'ties' | 'dare'
    finetuned_models: list,          # [clone_model, search_model]
    base_model: nn.Module,           # CodeBERT 基座
    scaling_coefficients: list = None, # [0.5, 0.5] 缩放系数
    # DARE/mask_merging 专用参数
    weight_mask_rate: float = 0.1,
    mask_strategy: str = "random",
    mask_apply_method: str = "task_arithmetic",
    # TIES 专用参数
    param_value_mask_rate: float = 0.8,
) -> dict:
    """执行模型合并
    
    内部流程:
      1. 创建 MergingMethod(merging_method_name=MERGE_METHODS[method])
      2. 调用 merging_method.get_merged_model(
           merged_model, models_to_merge,
           exclude_param_names_regex=[".*classifier.*", ".*mlp.*"],
           ...)
      3. 返回 merged_params (Dict[str, Tensor])
    
    排除规则: 合并仅针对共享的 encoder 参数
              任务特定头 (classifier / mlp) 保持各自微调后的权重
    """

def merge_and_evaluate(
    method: str,
    scaling_coefficients: list = None,
    device: str = "cuda",
    **merge_kwargs,
) -> dict:
    """合并 + 评测一体化调用
    
    流程:
      1. load_base_model()
      2. load_finetuned_model('clone_detection', ...) + load_finetuned_model('code_search', ...)
      3. merge_models(method, [model1, model2], base, ...)
      4. 将 merged_params 分别加载到两个任务模型中
      5. evaluate_clone_detection() + evaluate_code_search()
    
    返回: {
        'method': str,
        'clone_detection': {eval_f1, eval_precision, eval_recall},
        'code_search': {acc, precision, recall, f1},
        'scaling_coefficients': list,
    }
    """

def get_available_methods() -> list[dict]:
    """返回所有可用合并方法及其默认参数，供前端展示"""
```

**关键实现细节**:
- `MergingMethod` 类通过 `merging_method_name` 选择算法，所有方法共享 `get_merged_model()` 入口
- `exclude_param_names_regex=[".*classifier.*", ".*mlp.*"]` 排除任务特定头
- Task Arithmetic: `merged = base + Σ(α_i × (ft_i - base))`
- TIES: 先按幅度掩码 (mask_rate=0.8)，再按符号一致性投票，最后平均
- DARE (mask_merging): 随机/幅度掩码后再交给其他方法合并，支持 rescale
- 合并后需分别加载到克隆检测模型和代码搜索模型中评测（因任务头不同）

**合并方法映射** (前端显示名 → MergingMethod 内部名):

| 前端显示 | MergingMethod 参数 | 关键超参 | 说明 |
|----------|-------------------|----------|------|
| Task Arithmetic | `task_arithmetic` | `scaling_coefficients` | merged = base + Σ(α_i × τ_i) |
| TIES | `ties_merging` | `param_value_mask_rate` | 幅度掩码 + 符号投票 |
| DARE | `mask_merging` | `weight_mask_rate`, `mask_strategy`, `mask_apply_method` | 掩码 + 嵌套合并 |
| ModularEvo | — | — | 本文方法，需新增实现 |

**预置资源路径** (均位于 `demo_system/data/` 下的本地副本):

| 资源 | demo_system 内路径 | 原始来源 |
|--------|----------|----------|
| CodeBERT 预训练基座 | `data/models/codebert-base/` | `Tran_SeaM/data/pretrain_model/codebert-base/` |
| Java 模块 mask | `data/models/module_java/` | `Tran_SeaM/data/module_java/lr_0.001_alpha_10.0_ne_4_wrr_22.94/result/` |
| Python 模块 mask | `data/models/module_python/` | `Tran_SeaM/data/module_python/lr_0.001_alpha_10.0_ne_4_wrr_24.15/result/` |
| 克隆检测微调模型 | `data/models/finetuned_clone/` | `Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/.../checkpoint-best-f1/` |
| 代码搜索微调模型 | `data/models/finetuned_search/` | `Tran_SeaM/NL_code_search_WebQuery/code/save_model/.../checkpoint-best-aver/` |
| 克隆检测测试数据 | `data/datasets/clone_detection/` | `Tran_SeaM/Clone_detection_BigCloneBench_2/dataset/{test.txt, data.jsonl}` |
| 代码搜索验证数据 | `data/datasets/code_search/` | `Tran_SeaM/NL_code_search_WebQuery/CoSQA/cosqa_dev.json` |

### 7.2 第四章适配 (algorithm/chapter4/)

从 `TransModular_GPT/router/` 复制核心代码到 `algorithm/chapter4/libs/`，模型和数据存放在 `data/models/` 下。

```python
class Chapter4Adapter:
    def load_resources(self) -> Status       # 加载基座模型、任务向量、Router
    def get_baseline(self) -> dict           # 获取基线性能
    def infer_router(self, dataset) -> dict  # Router 推理，返回 α 系数
    def merge_and_evaluate(self, dataset, alphas) -> EvalResult  # 合并+评测
```

**代码依赖** (已复制到 `algorithm/chapter4/libs/` 下):

| 本地文件 | 原始来源 | 功能 |
|----------|----------|------|
| `libs/config.py` | `router/config.py` | 路径配置 (已改为本地路径) |
| `libs/router.py` | `router/router.py` | Router 网络定义 + 推理 |
| `libs/merge.py` | `router/merge.py` | 动态合并 `merged = base + Σ(α_i × τ_i)` |
| `libs/task_vectors.py` | `router/task_vectors.py` | 加载预计算任务向量 |
| `libs/data.py` | `router/data.py` | 用户上传数据的预处理 |

**预置模型路径** (均位于 `demo_system/data/` 下的本地副本):

| 资源 | demo_system 内路径 | 原始来源 |
|--------|----------|----------|
| GPT-Neo 125M 基座 | `data/models/gpt-neo-125m/` | `TransModular_GPT/data/gpt-neo-125m/` |
| 任务向量 (4个 .pt) | `data/models/task_vectors/` | `TransModular_GPT/router/data/task_vectors/` |
| 分类头 (4个 .pt) | `data/models/heads/` | `TransModular_GPT/router/data/heads/` |
| Router checkpoint | `data/models/router_checkpoint/` | `TransModular_GPT/router/checkpoints/20260303_...cosLR/` |
| 元数据 | `data/models/router_meta.pt` | `TransModular_GPT/router/data/meta.pt` |

**基线准确率** (来源: `evaluate.py` 硬编码):

| 任务 | 基线 ACC |
|------|----------|
| Code | 83.90% |
| LangID | 91.73% |
| Law | 70.57% |
| Math | 95.85% |

### 7.3 用户上传数据格式适配

用户上传统一格式 (`text + label + task_id`)，但各任务的原始数据加载器字段名不同。适配层负责转换：

```python
# 统一格式 → 各任务原始格式的字段映射
FIELD_MAPPING = {
    0: {"text": "func",    "label": "label"},    # code (parquet)
    1: {"text": "sentence", "label": "language"},  # langid (csv)
    2: {"text": "text",    "label": "label"},      # law/scotus (parquet)
    3: {"text": "text",    "label": "label"},      # math (parquet)
}

def convert_upload_to_task_format(records: list[dict]) -> dict[int, list[dict]]:
    """ 按 task_id 拆分用户上传数据，映射到各任务原始字段名 """
    ...
```

### 7.4 路径配置策略 — 自包含设计

**核心原则**: `demo_system/` 是完全自包含的可运行单元，所有模型、数据集、依赖代码均存放在其内部，**不引用任何外部文件夹** (`Tran_SeaM/`、`TransModular_GPT/` 等)。

```python
# config.py 统一使用 demo_system/ 内部的相对路径
import os
DEMO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")  # → demo_system/
DATA_ROOT = os.path.join(DEMO_ROOT, "data")         # → demo_system/data/
MODEL_ROOT = os.path.join(DATA_ROOT, "models")      # → demo_system/data/models/
DATASET_ROOT = os.path.join(DATA_ROOT, "datasets")   # → demo_system/data/datasets/
```

**资源准备方式**: 首次部署时运行 `setup_resources.sh` 脚本，从原始项目目录复制所需文件到 `demo_system/data/` 下（参见 7.5 节资源复制清单）。复制完成后 `demo_system/` 即可独立运行。

- 开发时: 运行一次 `setup_resources.sh` 后即可独立使用
- 部署迁移时: 打包整个 `demo_system/` 目录即可
- Docker 部署时: Dockerfile 中 `COPY demo_system/ /app/` 即为完整系统

### 7.5 资源复制清单 (setup_resources.sh)

> 首次部署时运行此脚本，从原始项目目录复制所需的模型、数据集和代码文件。

#### 7.5.1 模型文件复制

| 目标路径 (`demo_system/` 内) | 来源路径 | 关键文件 | 预估大小 |
|-----|------|------|------|
| `data/models/codebert-base/` | `Tran_SeaM/data/pretrain_model/codebert-base/` | config.json, pytorch_model.bin, tokenizer.json, vocab.json, merges.txt | ~500MB |
| `data/models/module_java/` | `Tran_SeaM/data/module_java/lr_0.001_alpha_10.0_ne_4_wrr_22.94/result/` | pytorch_model.bin, config.json | ~500MB |
| `data/models/module_python/` | `Tran_SeaM/data/module_python/lr_0.001_alpha_10.0_ne_4_wrr_24.15/result/` | pytorch_model.bin, config.json | ~500MB |
| `data/models/finetuned_clone/` | `Tran_SeaM/Clone_detection_BigCloneBench_2/code/saved_models/module_fintune_wrr_22.94_20250228/checkpoint-best-f1/` | pytorch_model.bin | ~500MB |
| `data/models/finetuned_search/` | `Tran_SeaM/NL_code_search_WebQuery/code/save_model/model_cosqa_20241031_epoch10/checkpoint-best-aver/` | pytorch_model.bin | ~500MB |
| `data/models/gpt-neo-125m/` | `TransModular_GPT/data/gpt-neo-125m/` | config.json, pytorch_model.bin, tokenizer files | ~500MB |
| `data/models/task_vectors/` | `TransModular_GPT/router/data/task_vectors/` | code.pt, langid.pt, law.pt, math.pt | ~2GB |
| `data/models/heads/` | `TransModular_GPT/router/data/heads/` | code.pt, langid.pt, law.pt, math.pt | ~1MB |
| `data/models/router_checkpoint/` | `TransModular_GPT/router/checkpoints/20260303_172150_.../` | checkpoint 文件 | ~1MB |
| `data/models/router_meta.pt` | `TransModular_GPT/router/data/meta.pt` | meta.pt | <1MB |

#### 7.5.2 数据集复制

| 目标路径 | 来源路径 | 关键文件 |
|------|------|------|
| `data/datasets/clone_detection/` | `Tran_SeaM/Clone_detection_BigCloneBench_2/dataset/` | test.txt, data.jsonl |
| `data/datasets/code_search/` | `Tran_SeaM/NL_code_search_WebQuery/CoSQA/` | cosqa_dev.json |
| `data/datasets/sample_datasets/` | 已存在 | mixed_sample_50.csv |

#### 7.5.3 代码文件复制

**第三章 libs/** (复制到 `algorithm/chapter3/libs/`):

| 目标文件 | 来源文件 | 是否需要修改 import | 说明 |
|------|------|------|------|
| `mask_layer.py` | `Tran_SeaM/mask_layer.py` | ❌ 无本地依赖 | MaskLinear, Binarization, init_mask_model |
| `sparse_utils.py` | `Tran_SeaM/utils.py` | ✅ `from mask_layer` → `from .mask_layer` | load_init_module_sparse() |
| `clone_model.py` | `Tran_SeaM/task_merge/task_eval/code_clone_eval.py` | ❌ 无本地依赖 | Model (CloneModel) + evaluate() |
| `search_model.py` | `Tran_SeaM/task_merge/task_eval/nl_code_search_eval.py` | ❌ 无本地依赖 | Model (SearchModel) + evaluate() |
| `merge_utils.py` | `Tran_SeaM/task_merge/merge_utils/merge_utils.py` | ❌ 无本地依赖 | get_param_names_to_merge() 等公共工具 |
| `task_vector.py` | `Tran_SeaM/task_merge/merge_methods/task_vector.py` | ✅ `from merge_utils.merge_utils` → `from .merge_utils` | TaskVector 类 |
| `mask_weights_utils.py` | `Tran_SeaM/task_merge/merge_methods/mask_weights_utils.py` | ✅ 修改本地 import 为相对导入 | mask_model_weights() |
| `merging_methods.py` | `Tran_SeaM/task_merge/merge_methods/merging_methods.py` | ✅ 修改本地 import 为相对导入 | MergingMethod 类 |

**第四章 libs/** (复制到 `algorithm/chapter4/libs/`):

| 目标文件 | 来源文件 | 是否需要修改 import | 说明 |
|------|------|------|------|
| `config.py` | `TransModular_GPT/router/config.py` | ✅ 路径改为指向 `data/models/` | 路径常量 |
| `router.py` | `TransModular_GPT/router/router.py` | ✅ `from config` → `from .config` | Router 网络 |
| `merge.py` | `TransModular_GPT/router/merge.py` | ✅ `from config` → `from .config` | 动态合并 |
| `task_vectors.py` | `TransModular_GPT/router/task_vectors.py` | ✅ `from config` → `from .config` | 任务向量加载 |
| `data.py` | `TransModular_GPT/router/data.py` | ✅ `from config` → `from .config` | 数据预处理 |

---

## 八、关键实现细节

### 8.1 耗时任务处理

第一版采用快速演示模式（加载预训练结果），大部分操作可同步完成。第四章评测（200 条数据推理）可能需要数十秒，使用 BackgroundTasks 异步执行。

- **当前方案**: FastAPI `BackgroundTasks` (轻量，适合 Demo)
- **预留扩展**: 保留 Celery + Redis 的架构接口，后续如开放实时训练可切换

### 8.2 模型加载与缓存

- **按需加载、用完释放**: 页面 A 和页面 B 的模型不同时常驻 GPU
- 进入某个章节页面时加载对应模型，切换页面时释放上一组
- 使用 LRU 缓存策略管理已加载的微调模型
- GPU 显存管理：同一时间只加载一个章节所需的模型到 GPU

### 8.3 前端实时进度

- **WebSocket**: `/ws/task/{task_id}` 推送训练日志和进度百分比
- **备选**: SSE (Server-Sent Events) 或短轮询

### 8.4 数据集格式规范 (第四章用户上传)

```json
// JSON 格式 — 4 个任务的混合数据
[
  {"text": "def hello(): print('hello')", "label": 3, "task_id": 0},
  {"text": "Bonjour le monde", "label": 2, "task_id": 1},
  {"text": "The court held that...", "label": 5, "task_id": 2},
  {"text": "If x + 3 = 7, find x", "label": 12, "task_id": 3}
]
```

```csv
# CSV 格式
text,label,task_id
"def hello(): print('hello')",3,0
"Bonjour le monde",2,1
"The court held that...",5,2
"If x + 3 = 7, find x",12,3
```

- 任务标识 (task_id): code(0) / langid(1) / law(2) / math(3)
- 最大样本数: 200 条（超出自动截断，UI 中不显性展示此限制）
- Token 长度限制: 与 GPT-Neo tokenizer max_length 一致

### 8.5 示例数据集

Demo 系统预置一个示例 CSV，用户可在页面 B 一键下载并直接上传体验：

- 文件: `data/sample_datasets/mixed_sample_50.csv`
- 内容: 从 4 个任务的测试集中各抽取 ~12 条，共 50 条混合数据
- 格式: `text,label,task_id`
- 前端在上传区域提供"下载示例数据"按钮

---

## 九、部署方案

### 9.1 开发环境

```bash
# 后端
cd backend
pip install fastapi uvicorn sqlalchemy torch transformers
uvicorn main:app --reload --port 8000

# 前端
cd frontend
npm install
npm run dev -- --port 3000
```

### 9.2 生产部署 (实验室集群 Docker 容器)

- Docker Compose 一键部署
- Nginx 反向代理 (前端静态文件 + API 转发)
- GPU 支持: NVIDIA Docker Runtime
- 容器内同时运行前后端服务

### 9.3 硬件要求

| 资源 | 最低要求 | 推荐配置 |
|------|----------|----------|
| GPU | 1× RTX 3060 (12GB) | 1× RTX 3090 (24GB) |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 50 GB (模型+数据) | 100 GB |

---

## 十、开发计划 & 里程碑

> 每个阶段设计为可独立验证的最小交付单元。完成每一步后可执行对应验证方式确认正确性。

### P0: 项目初始化 & 基础框架搭建

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P0.1 | 创建项目目录结构 (backend/frontend/algorithm/data) | `tree demo_system/` 确认目录结构完整 |
| P0.2 | FastAPI 后端空壳: main.py + CORS + 健康检查接口 | `curl http://localhost:8000/health` 返回 `{"status": "ok"}` |
| P0.3 | Vue 3 + Vite + Element Plus 前端空壳: 路由配置 (首页/第三章/第四章) + 首页布局 (简介+两张入口卡片) | 浏览器访问 `http://localhost:3000`，首页显示两张卡片，点击可跳转 |
| P0.4 | SQLite 数据库初始化 (tasks/models/evaluations 三张表) | 启动后端后 `sqlite3 demo.db ".tables"` 确认三张表存在 |
| P0.5 | 准备示例数据集 `data/sample_datasets/mixed_sample_50.csv` | `wc -l mixed_sample_50.csv` 确认 50 条，含 4 种 task_id |
| P0.6 | 一键启停脚本 (start.sh / stop.sh)，已在容器内运行无需 Docker | `bash start.sh` 可同时启动前后端服务，`curl localhost:8000/health` + 浏览器 `localhost:3000` 确认 |

### P1: 第三章算法适配层 (`algorithm/chapter3/`)

> 代码组织在 `demo_system/algorithm/chapter3/` 目录下，按职责拆分为 5 个文件。

| 步骤 | 文件 | 内容 | 验证方式 |
|------|------|------|----------|
| P1.1 | `config.py` | 路径配置：CODEBERT_PATH / MODULE_PATHS / FINETUNED_PATHS / EVAL_DATA_PATHS / MERGE_METHODS 常量 | ✅ 所有路径 exists=True |
| P1.2 | `model_loader.py` | `load_base_model()` — 加载 CodeBERT 基座 + tokenizer + config | ✅ RobertaModel 124.6M params |
| P1.3 | `model_loader.py` | `load_sparse_module(lang)` — 加载预训练 mask，返回稀疏率 + 逐层统计 | ✅ Java: sparsity=77.06%, WRR=22.94%, 144 layers |
| P1.4 | `model_loader.py` | `load_finetuned_model(task)` — 加载微调后的克隆检测/代码搜索模型 | ✅ Clone=125.8M, Search=127.0M |
| P1.5 | `evaluator.py` | `evaluate_clone()` — 封装克隆检测评测 | ✅ 返回 {eval_f1, eval_precision, eval_recall} |
| P1.6 | `evaluator.py` | `evaluate_search()` — 封装代码搜索评测 | ✅ 返回 {acc, precision, recall, f1} |
| P1.7 | `merger.py` | `merge_models(method, ...)` — 封装 3 种合并方法 | ✅ task_arithmetic 返回 199 keys |
| P1.8 | `merger.py` | `merge_and_evaluate(method)` — 合并 + 双任务评测一体化 | ✅ 流程验证通过 |
| P1.9 | 集成验证 | 导入、配置、GPU 加载、合并流程全部验证 | ✅ All tests PASSED |

### P2: 第三章后端 API & 前端页面 A

| 步骤 | 内容 | 验证方式 | 状态 |
|------|------|----------|------|
| P2.1 | 后端 API: `/api/ch3/modules` 返回可用模块列表 | `curl /api/ch3/modules` 返回 Java/Python 模块信息 JSON | ✅ |
| P2.2 | 后端 API: `/api/ch3/load-module` 加载指定模块 + `/api/ch3/finetuned` 微调模型信息 + `/api/ch3/evaluate/{task}` 评测 | `curl -X POST /api/ch3/load-module` 返回模块稀疏率 77.06%、WRR 22.94%、144 层逐层统计 | ✅ |
| P2.3 | 后端 API: `/api/ch3/merge` 执行合并 + 评测 | `curl -X POST /api/ch3/merge` 支持 task_arithmetic/ties/dare 三种方法 | ✅ |
| P2.4 | 后端 API: `/api/ch3/graph` 返回知识图谱数据 | `curl /api/ch3/graph` 返回 6 节点 + 6 条边 JSON | ✅ |
| P2.5 | 前端页面 A — Step 1 区域: 模块选择与加载，ECharts 柱状图展示逐层权重保留率 | 页面点击"加载 Java 模块"→ 显示逐层 bar chart + 稀疏率/WRR 标签 | ✅ |
| P2.6 | 前端页面 A — Step 2 区域: 展示预微调模型的评测结果 | 页面显示克隆检测 F1 和代码搜索 Precision | ✅ |
| P2.7 | 前端页面 A — Step 3 区域: 合并方法选择 + 结果对比图表 | 选择合并方法 → 配置系数 → 点击合并 → 显示准确率对比柱状图 | ✅ |

> **P2 实现文件清单:**
> - `backend/api/ch3_schemas.py` — Pydantic 请求/响应模型
> - `backend/api/chapter3.py` — 6 个 API 端点 (modules/load-module/finetuned/evaluate/merge/graph)
> - `frontend/src/api/chapter3.js` — Axios API 封装层
> - `frontend/src/views/Chapter3View.vue` — 完整页面 (知识图谱 + 三步流程)

### P3: 第四章算法适配层

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P3.1 | chapter4_adapter.py: `load_resources()` — 加载 GPT-Neo + 4 个任务向量 + Router | 调用返回 Status (各组件加载状态 + 参数量) |
| P3.2 | chapter4_adapter.py: `get_baseline()` — 获取 4 个单任务基线准确率 | 返回 `{code: 83.90, langid: 91.73, law: 70.57, math: 95.85}` |
| P3.3 | chapter4_adapter.py: `infer_router()` — 对混合数据集推理 α 系数 | 输入 50 条混合样本，返回 4 维 α 向量 |
| P3.4 | chapter4_adapter.py: `merge_and_evaluate()` — 合并+评测全流程 | 输入混合数据集，返回各任务准确率 + α 系数 |

### P4: 第四章后端 API & 前端页面 B

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P4.1 | 后端 API: `/api/ch4/status` 返回预部署模型状态 | `curl /api/ch4/status` 返回各组件加载状态 |
| P4.2 | 后端 API: `/api/ch4/baseline` 返回基线性能 | `curl /api/ch4/baseline` 返回 4 任务准确率 |
| P4.3 | 后端 API: `/api/ch4/upload-dataset` + `/api/ch4/evaluate` | 上传 CSV → 触发评测 → 轮询获取结果 |
| P4.4 | 后端 API: `/api/ch4/graph` 返回知识图谱数据 | `curl /api/ch4/graph` 返回 nodes + edges JSON |
| P4.5 | 前端页面 B — 左侧: 预部署模型信息卡片 | 页面加载后显示 GPT-Neo + 4 任务模型 + Router 状态 |
| P4.6 | 前端页面 B — 右侧: 数据集上传 + 评测触发 | 上传 CSV 文件 → 点击评测 → 显示加载动画 |
| P4.7 | 前端页面 B — 评测结果: α 系数图 + 准确率对比表 | 评测完成后展示 α 柱状图 + Merged vs. Baseline 表格 |

### P5: 知识图谱组件

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P5.1 | KnowledgeGraph.vue 通用组件: ECharts 关系图渲染 | 传入 mock 数据，页面渲染出节点和边 |
| P5.2 | 集成到页面 A: 第三章知识图谱 (6 节点 + 对应边) | 页面 A 顶部显示完整图谱，Tooltip 可交互 |
| P5.3 | 集成到页面 B: 第四章知识图谱 (含 Router 节点) | 页面 B 顶部显示完整图谱，评测后动态更新 |

### P6: 系统集成 & 部署

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P6.1 | 前后端联调: 全流程走通 (页面 A 三步 + 页面 B 上传评测) | 手动操作全流程无报错，结果数据正确 |
| P6.2 | Docker 镜像打包 (含模型文件 + 前端构建产物) | `docker-compose up` 启动后可访问完整 Demo |
| P6.3 | 实验室集群部署 & GPU 验证 | 集群容器中运行，GPU 推理速度正常 |
| P6.4 | 用户验收: 完整演示流程录屏 | 录制 3-5 分钟演示视频，覆盖所有核心功能 |

---

## 十一、需求确认记录

> 以下为历次沟通确认的需求决策，作为后续开发的依据。

### 第一轮确认 (2026-03-06)

#### 第三章页面 (页面 A)

| # | 问题 | 决策 |
|---|------|------|
| 1 | 模块化训练模式 | **快速演示模式**（加载预训练 mask），预留训练接口 |
| 2 | 下游任务范围 | **两个都要**: 克隆检测 + 代码搜索（合并需要 ≥2 任务） |
| 3 | 合并方法 | **4 种**: Task Arithmetic / TIES / DARE / ModularEvo |
| 4 | 语言模块 | **2 种**: Java（克隆检测）、Python（代码搜索） |

#### 第四章页面 (页面 B)

| # | 问题 | 决策 |
|---|------|------|
| 5 | 上传数据集标签 | **4 个已知任务的混合数据** |
| 6 | 评测展示深度 | **聚合指标为主**，后续扩展前 10 样本展示 |
| 7 | 数据集大小限制 | **最多 200 条**，UI 不显性展示 |

#### 通用设计

| # | 问题 | 决策 |
|---|------|------|
| 8 | 知识图谱边类型 | **4 种**: 微调、模块化、模块微调、合并 |
| 9 | 用户认证 | **不需要**，内部展示系统 |
| 10 | 部署环境 | **实验室集群 Docker 容器** |
| 11 | UI 组件库 | **Element Plus**（简洁美观） |
| 12 | 参考图片 | 前期设计原型图 (demo1/2/3.png)，适当参考 |

### 第二轮确认 (2026-03-06)

#### 算法 & 指标修正

| # | 问题 | 决策 |
|---|------|------|
| 13 | DARE 合并方法对应关系 | **DARE = 代码中的 `mask_merging`**；新增 **ModularEvo** 作为本文方法 |
| 14 | 代码搜索评测指标 | 使用 **Precision**（非 MRR），与实际代码 `nl_code_search_eval.py` 一致 |
| 15 | 第四章基线准确率 | **Code: 83.90%** / LangID: 91.73% / Law: 70.57% / Math: 95.85% |

#### 技术实现细节

| # | 问题 | 决策 |
|---|------|------|
| 16 | 用户上传数据格式适配 | **写适配层**: 统一 `text+label+task_id` → 按 task_id 拆分映射到各任务原始字段 |
| 17 | Router checkpoint 选择 | `checkpoints/20260303_172150_bs32_lr0.001_ep5_bpe1000_lam0.5_dir0.3_alp1.0_cosLR/` |
| 18 | 预微调模型选择 | 克隆检测: `module_fintune_wrr_22.94_20250228/checkpoint-best-f1/`；代码搜索: `model_cosqa_20241031_epoch10/checkpoint-best-aver/` |
| 19 | 模型路径配置 | **环境变量 + 相对路径**，`DEMO_PROJECT_ROOT` / `DEMO_ALGO_ROOT` |
| 20 | GPU 显存策略 | **按需加载、用完释放**，不同时常驻 |
| 21 | 示例数据集 | **提供预置 CSV**，50 条混合数据，前端一键下载 |
| 22 | 首页内容 | **系统简介 + 两个章节入口卡片** |

