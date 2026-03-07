# LLMOps Platform — 系统开发手册

> **版本**: v0.4 (UI 重构版)  
> **更新日期**: 2026-03-07  
> **技术栈**: FastAPI + Vue 3 + Element Plus + ECharts + SQLite  

---

## 一、系统概述

### 1.1 项目目标

构建一个用于**展示 Transformer 模型模块化与动态合并**算法的 Demo 系统，涵盖论文第三章（基于 CodeBERT 的模块化方法）与第四章（基于 GPT-Neo 的 Router 动态合并方法）的核心流程，以可视化、可交互的方式呈现给用户。

### 1.2 核心展示内容

| 页面 | 算法名称 | 基座模型 | 展示流程 |
|------|----------|----------|----------|
| ModularEvo 进化 | 第三章 — 模块化进化 | CodeBERT (codebert-base) | 模块化 → 稀疏微调 → 知识组合（模型合并） |
| AutoRouter 自动组合 | 第四章 — Router 动态合并 | GPT-Neo 125M | 输入识别 → 权重组合 → 自动匹配分类头 |

### 1.3 全局 UI 框架

- **布局风格**: 经典 Admin Dashboard — 左侧固定侧栏 + 顶部固定导航 + 右侧主内容滚动区
- **主题配色**: 浅色主题，顶部导航栏为深紫色
- **UI 组件库**: Element Plus (Vue 3)
- **无需用户认证**: 内部展示系统

---

## 二、系统架构

### 2.1 整体架构图

```
┌──────────────────────────────────────────────────────────────┐
│ 顶部导航栏 (深紫色)                                          │
│ [Logo: LLMOps Platform] 首页 | 模型介绍 | ModularEvo | AutoRouter  [🟢 系统正常] │
├──────────┬───────────────────────────────────────────────────┤
│ 左侧栏   │ 主内容区 (滚动)                                   │
│ ┌──────┐ │                                                   │
│ │运行   │ │  根据导航切换显示不同页面:                         │
│ │任务   │ │  - 首页: 算法介绍 + 入口卡片                     │
│ │      │ │  - 模型介绍: 各模型详情                           │
│ │ModEvo│ │  - ModularEvo: 6 行功能卡片                      │
│ │Router│ │  - AutoRouter: 4 行功能卡片                      │
│ ├──────┤ │                                                   │
│ │系统   │ │                                                   │
│ │状态   │ │                                                   │
│ │      │ │                                                   │
│ │GPU   │ │                                                   │
│ │MEM   │ │                                                   │
│ └──────┘ │                                                   │
└──────────┴───────────────────────────────────────────────────┘
                   │ REST API
┌──────────────────▼──────────────────────────┐
│               后端 (FastAPI)                 │
│  ┌──────────────────────────────────────┐   │
│  │ API 层: 系统状态 / 模型管理 / 评测   │   │
│  ├──────────────────────────────────────┤   │
│  │ 算法层: chapter3 / chapter4          │   │
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
│   ├── main.py                  # 应用入口 (CORS + 根路径重定向 /docs)
│   ├── api/                     # API 路由
│   │   ├── chapter3.py          # ModularEvo 相关接口 (✅ 已有)
│   │   ├── ch3_schemas.py       # ModularEvo Pydantic 模型 (✅ 已有)
│   │   ├── chapter4.py          # AutoRouter 相关接口
│   │   ├── ch4_schemas.py       # AutoRouter Pydantic 模型
│   │   └── system.py            # 系统状态接口 (GPU/内存)
│   ├── models/                  # 数据库模型 (SQLAlchemy)
│   │   ├── database.py          # 数据库初始化
│   │   └── schemas.py           # 公共 Pydantic 模型
│   └── core/
│       └── config.py
├── frontend/                    # Vue 3 前端
│   ├── src/
│   │   ├── App.vue              # 根组件 (Admin Dashboard 布局框架)
│   │   ├── main.js              # 入口
│   │   ├── style.css            # 全局样式 (深紫色主题变量)
│   │   ├── router/
│   │   │   └── index.js         # 路由: / /models /modularevo /autorouter
│   │   ├── views/
│   │   │   ├── HomeView.vue          # 首页 (算法介绍 + 入口卡片)
│   │   │   ├── ModelsView.vue        # 模型介绍页
│   │   │   ├── ModularEvoView.vue    # ModularEvo 进化页 (6 行卡片)
│   │   │   └── AutoRouterView.vue    # AutoRouter 自动组合页 (4 行卡片)
│   │   ├── components/
│   │   │   ├── AppHeader.vue         # 顶部导航栏 (深紫色)
│   │   │   ├── AppSidebar.vue        # 左侧侧边栏 (运行任务+系统状态)
│   │   │   ├── KnowledgeGraph.vue    # 知识图谱组件 (ECharts 关系图)
│   │   │   └── TerminalLog.vue       # 终端日志组件 (黑底绿字)
│   │   └── api/                 # Axios 接口封装
│   │       ├── index.js
│   │       ├── chapter3.js      # ModularEvo API (✅ 已有)
│   │       ├── chapter4.js      # AutoRouter API
│   │       └── system.js        # 系统状态 API
│   └── ...
├── algorithm/                   # 算法适配层 (自包含)
│   ├── chapter3/                # ModularEvo 适配 (✅ P1 已完成)
│   │   ├── config.py / model_loader.py / evaluator.py / merger.py
│   │   └── libs/                # 核心依赖代码
│   └── chapter4/                # AutoRouter 适配
│       ├── adapter.py
│       └── libs/                # Router 核心代码
├── data/                        # 所有模型和数据集 (自包含, ✅ 已迁移)
│   ├── models/ / datasets/
└── start.sh / stop.sh           # 一键启停脚本
```

### 2.4 全局布局设计

#### 2.4.1 顶部导航栏 (AppHeader)

```
┌────────────────────────────────────────────────────────────────────┐
│ [Logo] LLMOps Platform  🏠首页 📊模型介绍 🧬ModularEvo 🔀AutoRouter  🟢系统正常 │
└────────────────────────────────────────────────────────────────────┘
```

- **背景色**: 深紫色 (`#5b21b6`)
- **左侧**: 平台 Logo + 名称 "LLMOps Platform" (白色文字)
- **中部**: 水平导航菜单 (`el-menu` mode=horizontal, 白色文字/下划线高亮)
  - 🏠 首页 (路由 `/`)
  - 📊 模型介绍 (路由 `/models`)
  - 🧬 ModularEvo 进化 (路由 `/modularevo`)
  - 🔀 AutoRouter 自动组合 (路由 `/autorouter`)
- **右侧**: 🟢 小绿点 + "系统正常" 文字

#### 2.4.2 左侧侧边栏 (AppSidebar)

宽度约 220px，固定在左侧，分两个卡片区块垂直堆叠:

**区块一: 运行任务 (Running Tasks)**
| 项目 | 当前页面时 | 非当前页面时 |
|------|----------|-------------|
| ModularEvo 进化 | 浅黄色背景 + ⚙️齿轮旋转动画 | 蓝色齿轮(静止) |
| AutoRouter 自动组合 | 浅黄色背景 + ⚙️齿轮旋转动画 | 蓝色齿轮(静止) |

**区块二: 系统状态 (System Status)**
- GPU 利用率: 右对齐数值 (如 `67%`)
- 内存使用: 右对齐数值 (如 `24GB/32GB`)

> 系统状态从后端 `/api/system/status` 接口轮询获取 (每 5 秒)

#### 2.4.3 首页 (HomeView)

```
┌───────────────────────────────────────────────────┐
│ LLMOps Platform                                   │
│ 大语言模型模块化与动态合并展示平台                │
│                                                   │
│ ┌─────────────────────┐ ┌─────────────────────┐   │
│ │ 🧬 ModularEvo 进化   │ │ 🔀 AutoRouter 组合   │   │
│ │ 基于 CodeBERT        │ │ 基于 GPT-Neo 125M   │   │
│ │ 模块化 → 独立进化 →  │ │ 输入识别 → 权重组合 │   │
│ │ 知识组合             │ │ → 自动匹配分类头     │   │
│ │ [进入 ▶]             │ │ [进入 ▶]             │   │
│ └─────────────────────┘ └─────────────────────┘   │
└───────────────────────────────────────────────────┘
```

---

## 三、ModularEvo 进化页面 (ModularEvoView)

### 3.1 页面总体布局

页面在主内容区内纵向排布 **6 行功能卡片** (`el-card`)，每行一个卡片，依次展示模块化进化的完整流程。

```
┌─────────────────────────────────────────────────────────┐
│ 卡片 1: 算法介绍 — 模块化 → 独立进化 → 知识组合        │
├─────────────────────────────────────────────────────────┤
│ 卡片 2: 知识图谱 — CodeBERT 模型关系可视化              │
├─────────────────────────────────────────────────────────┤
│ 卡片 3: 模型信息 — 左侧模型选择列表 / 右侧参数详情     │
├─────────────────────────────────────────────────────────┤
│ 卡片 4: 模块化 — 模块稀疏度、权重保留率展示             │
├─────────────────────────────────────────────────────────┤
│ 卡片 5: 稀疏微调 — 加载微调模型 + 性能评测展示          │
├─────────────────────────────────────────────────────────┤
│ 卡片 6: 模型合并 — 方法选择 + 参数配置 + 终端日志输出    │
└─────────────────────────────────────────────────────────┘
```

### 3.2 卡片详细设计

#### 3.2.1 卡片 1 — 算法介绍

- **内容**: 用简洁图文展示 ModularEvo 三步流程
- **布局**: 横向三栏 (模块化 → 独立进化 → 知识组合)，每栏图标 + 简短描述
- **纯展示**: 无交互，仅介绍算法思想

#### 3.2.2 卡片 2 — 知识图谱

- **组件**: `<KnowledgeGraph>` (ECharts 关系图)
- **数据**: 从 `GET /api/ch3/graph` 获取 6 节点 + 6 边
- **节点**: CodeBERT (预训练) → Module-Java / Module-Python (模块) → FT-CloneDet / FT-CodeSearch (微调) → Merged-Model (合并)
- **交互**: 悬停显示 Tooltip (参数量、稀疏率)

#### 3.2.3 卡片 3 — 模型信息

- **左侧**: 模型选择列表 (`el-menu` 或 `el-radio-group`)，列出知识图谱中涉及的所有模型:
  - CodeBERT (基座)
  - Module-Java / Module-Python (稀疏模块)
  - FT-CloneDet / FT-CodeSearch (微调模型)
  - Merged-Model (合并后模型，合并完成后才可选)
- **右侧**: 展示选中模型的详细信息
  - 参数量 (如 124.6M)
  - 模型类型 (预训练/模块/微调/合并)
  - 其他元信息 (如路径、来源)

#### 3.2.4 卡片 4 — 模块化

- **功能**: 展示预训练好的稀疏模块的统计信息
- **操作**: 选择语言模块 (Java / Python) → 点击"加载模块" → 调用 `POST /api/ch3/load-module`
- **展示**:
  - 模块稀疏率 (如 77.06%)、权重保留率 WRR (如 22.94%)、层数 (如 144 层)
  - 逐层权重保留率 ECharts 柱状图 (bar chart)

#### 3.2.5 卡片 5 — 稀疏微调 (模块微调)

- **功能**: 加载预微调好的模型并展示评测性能
- **操作**: 
  - 加载按钮: 加载已有的微调模型 checkpoint
  - 评测按钮: 调用 `POST /api/ch3/evaluate/{task}` (跑部分样本快速出结果，然后展示已有完整评测结果)
- **展示**:
  - 克隆检测: F1, Precision, Recall
  - 代码搜索: Accuracy, Precision, Recall, F1
  - 模型参数量 (Clone=125.8M, Search=127.0M)

#### 3.2.6 卡片 6 — 模型合并

- **顶部**: 合并方法选择 + 参数配置
  - 合并方法 (`el-select`): **Task Arithmetic** / **TIES Merging** / **DARE** / **ModularEvo**
  - 缩放系数输入框
  - TIES 专用: `param_value_mask_rate` 滑块
  - DARE 专用: `weight_mask_rate` 输入
  - ModularEvo: 自动加载稀疏微调模型 (实际实现 = 稀疏微调 + Task Arithmetic，但前端不暴露此细节，与其他三种方法并列展示)
  - 合并按钮: 调用 `POST /api/ch3/merge`

- **底部**: 黑底终端日志区 (`<TerminalLog>` 组件)
  - 视觉: 黑色背景 + 绿色/白色等宽字体，模仿终端
  - 内容: 实时输出合并进度日志 (加载模型 → 计算任务向量 → 执行合并 → 评测中 → 完成)
  - 合并完成后在日志区下方或旁边展示结果:
    - 各任务准确率对比表
    - 4 种方法性能对比柱状图 (如果已运行多种方法)

**ModularEvo 合并方法特殊处理**:
- 前端: 显示为独立方法 "ModularEvo"，与 Task Arithmetic / TIES / DARE 并列
- 后端: 实际调用 = 稀疏微调模型 + Task Arithmetic 合并
- 其他三种方法: 使用全微调模型

### 3.3 知识图谱数据 (第三章)

与已有 `GET /api/ch3/graph` 一致:

- **节点** (6 个): CodeBERT, Module-Java, Module-Python, FT-CloneDet, FT-CodeSearch, Merged-Model
- **边** (6 条): 模块化 ×2, 模块微调 ×2, 合并 ×2
- **交互**: 悬停 Tooltip 显示模型详情

### 3.4 后端 API 设计

> 已有 API (✅ P2 已实现，需在 UI 重构中适配前端调用方式):

```
GET    /api/ch3/modules              # 返回可用模块列表 (java/python)
POST   /api/ch3/load-module          # 加载模块，返回稀疏率+逐层统计
GET    /api/ch3/finetuned            # 返回微调模型信息
POST   /api/ch3/evaluate/{task}      # 评测指定任务
POST   /api/ch3/merge               # 执行合并+评测
GET    /api/ch3/graph                # 返回知识图谱数据
```

> 需新增/修改 API:

```
GET    /api/ch3/models               # 返回所有模型详细信息 (供卡片3使用)
POST   /api/ch3/merge               # 扩展: 支持 modularevo 方法 + 流式日志输出
```

---

## 四、AutoRouter 自动组合页面 (AutoRouterView)

### 4.1 页面总体布局

页面纵向排布 **4 行功能卡片**:

```
┌─────────────────────────────────────────────────────────┐
│ 卡片 1: 算法介绍 — 输入识别 → 权重组合 → 自动匹配分类头 │
├─────────────────────────────────────────────────────────┤
│ 卡片 2: 知识图谱 — GPT-Neo 模型关系可视化               │
├─────────────────────────────────────────────────────────┤
│ 卡片 3: 模型信息 — 左侧模型选择 / 右侧参数详情          │
├─────────────────────────────────────────────────────────┤
│ 卡片 4: AutoRouter 评测 — 数据集上传 + 终端日志 + 结果   │
└─────────────────────────────────────────────────────────┘
```

### 4.2 卡片详细设计

#### 4.2.1 卡片 1 — 算法介绍

- **布局**: 横向三栏 (输入识别 → 权重组合 → 自动匹配分类头)
- **纯展示**: 图标 + 简短描述，介绍 AutoRouter 算法流程

#### 4.2.2 卡片 2 — 知识图谱

- **组件**: `<KnowledgeGraph>` (复用 ModularEvo 同一组件，传入不同数据)
- **数据**: 从 `GET /api/ch4/graph` 获取
- **节点**: GPT-Neo 125M → FT-Code / FT-LangID / FT-Law / FT-Math → TaskVector ×4 → Router → Merged-Model
- **交互**: 悬停节点显示参数量、基线准确率

#### 4.2.3 卡片 3 — 模型信息

- **左侧**: 模型选择列表 (知识图谱中涉及的所有模型):
  - GPT-Neo 125M (基座, ~125M params)
  - FT-Code / FT-LangID / FT-Law / FT-Math (微调模型)
  - Router (~98,952 params)
  - 各 TaskVector
- **右侧**: 选中模型的详细信息 (参数量、类型、基线性能等)

#### 4.2.4 卡片 4 — AutoRouter 评测

- **顶部**: 数据集上传区域
  - 文件上传组件 (`el-upload`): 支持 CSV/JSON
  - "下载示例数据集" 按钮 (预置 `mixed_sample_50.csv`)
  - "开始评测" 按钮: 调用 `POST /api/ch4/evaluate`
  - 数据集格式要求提示: `text + label + task_id`

- **底部**: 左右两栏布局
  - **左侧**: 黑底终端日志区 (`<TerminalLog>`)
    - 实时输出评测流程日志 (加载模型 → Router 推理 → 计算 α → 合并 → 推理 → 完成)
  - **右侧**: 评测结果区 (评测结束后出现)
    - 任务分类准确率 (各 task 的 ACC)
    - 每任务 α 系数 (Router 学习到的权重分布，柱状图)
    - Merged vs. Baseline 对比表格
    - 任务识别准确率 & Recall 汇总

### 4.3 知识图谱数据 (第四章)

- **节点**: GPT-Neo 125M, FT-Code/LangID/Law/Math, TaskVector ×4, Router, Merged-Model
- **边类型**: 微调 (GPT-Neo → FT), 模块化 (FT → TaskVector), 合并 (TaskVector × N + Router → Merged)

### 4.4 后端 API 设计

```
GET    /api/ch4/status               # 预部署模型/Router 加载状态
GET    /api/ch4/baseline             # 4 任务基线准确率
GET    /api/ch4/models               # 所有模型详细信息 (供卡片3)
POST   /api/ch4/upload-dataset       # 上传自定义数据集
POST   /api/ch4/evaluate             # 启动评测 (返回流式日志)
GET    /api/ch4/evaluate/{task_id}   # 查询评测进度与结果
GET    /api/ch4/graph                # 知识图谱数据
```

### 4.5 公共后端 API

```
GET    /api/system/status            # 系统状态 (GPU 利用率、内存使用)
GET    /health                       # 健康检查
```


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

- **按需加载、用完释放**: ModularEvo 和 AutoRouter 页面的模型不同时常驻 GPU
- 进入某个功能页面时加载对应模型，切换页面时释放上一组
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

Demo 系统预置一个示例 CSV，用户可在 AutoRouter 页面一键下载并直接上传体验：

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

## 十、开发计划 & 里程碑 (v0.4 UI 重构版)

> 每个阶段设计为可独立验证的最小交付单元。✅ 表示已完成，后续阶段从 **P2 (UI 重构)** 开始。

### P0: 项目初始化 & 基础框架搭建 ✅

| 步骤 | 内容 | 状态 |
|------|------|------|
| P0.1 | 创建项目目录结构 (backend/frontend/algorithm/data) | ✅ |
| P0.2 | FastAPI 后端: main.py + CORS + 健康检查 + 根路径重定向 /docs | ✅ |
| P0.3 | Vue 3 + Vite + Element Plus 前端空壳 + 路由 | ✅ |
| P0.4 | SQLite 数据库初始化 | ✅ |
| P0.5 | 示例数据集 `mixed_sample_50.csv` | ✅ |
| P0.6 | 一键启停脚本 (start.sh / stop.sh) | ✅ |

### P1: 第三章算法适配层 ✅

| 步骤 | 文件 | 内容 | 状态 |
|------|------|------|------|
| P1.1 | `config.py` | 路径配置常量 | ✅ |
| P1.2 | `model_loader.py` | `load_base_model()` — CodeBERT 124.6M | ✅ |
| P1.3 | `model_loader.py` | `load_sparse_module()` — Java WRR=22.94%, 144层 | ✅ |
| P1.4 | `model_loader.py` | `load_finetuned_model()` — Clone=125.8M, Search=127.0M | ✅ |
| P1.5 | `evaluator.py` | `evaluate_clone()` | ✅ |
| P1.6 | `evaluator.py` | `evaluate_search()` | ✅ |
| P1.7 | `merger.py` | `merge_models()` — 3 种合并方法 | ✅ |
| P1.8 | `merger.py` | `merge_and_evaluate()` — 合并+评测一体化 | ✅ |
| P1.9 | 集成验证 | GPU 全流程验证 | ✅ |

### P1.5: 第三章后端 API ✅ (原 P2 后端部分)

> 后端 API 已完成，UI 重构不影响后端接口。

| 步骤 | 内容 | 状态 |
|------|------|------|
| P1.5.1 | `GET /api/ch3/modules` 返回 Java/Python 模块信息 | ✅ |
| P1.5.2 | `POST /api/ch3/load-module` 返回稀疏率+逐层统计 | ✅ |
| P1.5.3 | `GET /api/ch3/finetuned` 微调模型信息 | ✅ |
| P1.5.4 | `POST /api/ch3/evaluate/{task}` 评测 | ✅ |
| P1.5.5 | `POST /api/ch3/merge` 合并+评测 (task_arithmetic/ties/dare) | ✅ |
| P1.5.6 | `GET /api/ch3/graph` 知识图谱数据 (6节点+6边) | ✅ |

> **已有实现文件**: `backend/api/ch3_schemas.py`, `backend/api/chapter3.py`, `frontend/src/api/chapter3.js`

---

### ★ P2: UI 重构 — 全局框架 + 公共组件 ← 从这里开始

> 重构前端为 Admin Dashboard 布局，创建公共组件框架。

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P2.1 | **全局样式**: `style.css` 定义深紫色主题 CSS 变量 (#5b21b6 主色) | 浏览器检查样式变量生效 |
| P2.2 | **AppHeader.vue**: 深紫色顶部导航栏 (Logo + 4个导航项 + 系统状态指示器) | 页面顶部显示深紫色导航栏，导航可切换 |
| P2.3 | **AppSidebar.vue**: 左侧侧边栏 (运行任务区块 + 系统状态区块)，齿轮动画 | 侧边栏显示两个区块，切换页面时齿轮状态变化 |
| P2.4 | **App.vue 布局重构**: `el-container` Admin Dashboard 布局 (Header + Sidebar + Main) | 页面呈现三栏布局，主内容区可滚动 |
| P2.5 | **路由更新**: `router/index.js` 新增 `/models` `/modularevo` `/autorouter` 路由 | URL 切换正常，组件加载正确 |
| P2.6 | **TerminalLog.vue**: 黑底终端日志组件 (等宽字体, 自动滚动, 支持 props 传入日志行) | 组件渲染黑底绿字终端效果 |
| P2.7 | **KnowledgeGraph.vue**: 通用知识图谱组件 (接收 nodes/edges props, ECharts 关系图) | 传入 mock 数据渲染出图谱 |
| P2.8 | **系统状态后端 API**: `GET /api/system/status` 返回 GPU/内存信息 | `curl /api/system/status` 返回 JSON |
| P2.9 | **HomeView.vue**: 首页重写 (两个算法入口卡片 + 简介) | 首页显示两张卡片，点击可跳转 |

### P3: UI 重构 — ModularEvo 进化页面

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P3.1 | **ModularEvoView.vue 卡片1**: 算法介绍 (三栏: 模块化→独立进化→知识组合) | 卡片显示三步图文介绍 |
| P3.2 | **卡片2**: 知识图谱 (集成 `<KnowledgeGraph>`, 调用 `/api/ch3/graph`) | 页面显示 6 节点关系图 |
| P3.3 | **卡片3**: 模型信息 (左侧模型选择列表 + 右侧详情展示) | 点击模型名显示参数量等信息 |
| P3.4 | **卡片4**: 模块化 (选择语言模块+加载+稀疏率/WRR+逐层柱状图) | 加载 Java 模块显示 sparsity=77.06% + 柱状图 |
| P3.5 | **卡片5**: 稀疏微调 (加载微调模型+评测+性能指标展示) | 评测后显示 F1, Precision, Recall |
| P3.6 | **卡片6**: 模型合并 — 顶部方法选择(4种)+参数配置 | 选择方法后参数面板切换 |
| P3.7 | **卡片6**: 底部终端日志区 (`<TerminalLog>`) + 合并结果展示 | 点击合并→终端输出日志→显示结果 |
| P3.8 | **后端适配**: `/api/ch3/merge` 支持 modularevo 方法 + 日志流 | curl merge 传 modularevo 返回结果 |

### P4: 第四章算法适配层 (`algorithm/chapter4/`)

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P4.1 | `adapter.py`: `load_resources()` — 加载 GPT-Neo + 4 任务向量 + Router | 返回各组件加载状态 + 参数量 |
| P4.2 | `adapter.py`: `get_baseline()` — 4 个单任务基线准确率 | 返回 {code:83.90, langid:91.73, law:70.57, math:95.85} |
| P4.3 | `adapter.py`: `infer_router()` — Router 推理 α 系数 | 输入 50 条混合样本，返回 4 维 α |
| P4.4 | `adapter.py`: `merge_and_evaluate()` — 合并+评测 | 返回各任务准确率 + α 系数 |

### P5: 第四章后端 API + AutoRouter 页面

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P5.1 | 后端 API: `/api/ch4/status` `/api/ch4/baseline` `/api/ch4/graph` `/api/ch4/models` | curl 验证各 API 正常 |
| P5.2 | 后端 API: `/api/ch4/upload-dataset` + `/api/ch4/evaluate` | 上传 CSV → 触发评测 → 返回结果 |
| P5.3 | **AutoRouterView.vue 卡片1**: 算法介绍 (输入识别→权重组合→自动匹配分类头) | 卡片显示三步介绍 |
| P5.4 | **卡片2**: 知识图谱 (集成 `<KnowledgeGraph>`, 调用 `/api/ch4/graph`) | 显示 GPT-Neo 关系图 |
| P5.5 | **卡片3**: 模型信息 (模型选择+详情) | 点击模型显示参数量/基线 |
| P5.6 | **卡片4**: AutoRouter 评测 (上传+日志+结果) | 上传→终端日志→右侧出现 α 系数+准确率 |

### P6: 首页增强 + UI 美化 + 系统集成

> 去除独立模型介绍页 (ModelsView)，将算法介绍融入首页；删除导航栏"模型介绍"入口和 `/models` 路由。首页每个算法卡片增加方法框架图。全局 UI 美化。

| 步骤 | 内容 | 验证方式 |
|------|------|----------|
| P6.1 | **删除 ModelsView**: 移除 `/models` 路由、导航栏入口、ModelsView.vue 文件 | `/models` 路径返回 404，导航栏无"模型介绍" |
| P6.2 | **首页增强**: HomeView 每个卡片增加算法框架图 (`modularevo.png` / `router.png`)，图片放入 `frontend/public/images/` | 首页显示两张带图片的卡片 |
| P6.3 | **UI 美化**: (1) 首页 Hero 区域增加渐变背景和装饰  (2) 入口卡片改为左图右文的横向排版  (3) 各页面卡片标题增加渐变装饰条  (4) 全局滚动条美化、卡片间距/圆角/阴影统一 | 浏览器视觉审查，界面更精致 |
| P6.4 | 前后端联调: ModularEvo 全流程贯通 | 手动操作 6 张卡片无报错 |
| P6.5 | 前后端联调: AutoRouter 全流程贯通 | 上传数据集→评测→结果展示 |
| P6.6 | 系统集成验收 | 全功能流程无报错 |


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

