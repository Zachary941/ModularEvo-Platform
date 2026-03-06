# 基于双头 Router 的任务向量动态合并算法

## 1. 概述

本算法提出了一种**基于轻量级双头 Router 的权重集成模型自动协同复用方法**。核心思想是：给定一个预训练基座模型和若干个在不同任务上微调过的模型，**通过 Router 网络根据用户输入数据的分布，自动生成任务向量（Task Vector）的合并系数 α**，将多个微调模型的知识动态融合为**唯一一个主干网络**（Backbone），同时使用 Router 内置的**任务分类器**将每条样本路由到对应的冻结分类头，实现多任务统一推理。

**核心优势**：推理时只需加载**一个合并后的主干 + 多个轻量分类头**，而非多个完整微调模型，显著节省 VRAM。

---

## 2. 问题定义

**给定：**
- 预训练基座模型参数 $\theta_{\text{pre}}$（例如 GPT-Neo 125M）
- $K$ 个在不同任务上微调后的模型参数 $\{\theta_{\text{ft}}^{(k)}\}_{k=1}^{K}$
- 每个微调模型对应一个冻结的分类头 $H^{(k)}$
- 用户的目标数据集 $\mathcal{D}_{\text{target}}$（可能混合多个任务）

**目标：**
1. 学习一个 Router 网络 $R_\phi$，根据输入数据的分布自动输出合并系数 $\boldsymbol{\alpha} \in [0,1]^K$
2. 使用 $\boldsymbol{\alpha}$ 将多个任务向量合并为唯一主干：$\theta^* = \theta_{\text{pre}} + \sum_{k=1}^{K} \alpha_k \cdot \tau^{(k)}$
3. 同时训练 Router 内的任务分类器，对每条样本预测其所属任务，路由至对应分类头

---

## 3. Router 网络结构与设计

### 3.1 整体架构

Router 是一个**双头轻量级 MLP 网络**，共享一个冻结的 Embedding 编码器（来自预训练模型的 `transformer.wte`），具有两个功能头：

```
输入: input_ids [B, L], attention_mask [B, L]
            │
            ▼
┌────────────────────────────┐
│   冻结 Embedding (wte)      │  ← GPT-Neo 的 token embedding, 不训练
│   [B, L] → [B, L, 768]     │
└────────────────────────────┘
            │
            ▼
┌────────────────────────────┐
│   Mean Pooling + detach     │  排除 padding, 按 attention_mask 加权平均
│   [B, L, 768] → [B, 768]   │  → sample_embeds
└────────────────────────────┘
            │
       ┌────┴─────────────────────┐
       │                          │
       ▼                          ▼
┌──────────────────┐    ┌───────────────────────────────┐
│ Task Classifier  │    │        Merge Head              │
│  (Head 2)        │    │      (Head 1)                  │
│                  │    │                                │
│ 768→128→64→4     │    │ 输入 = [embed ⊕ task_probs]    │
│     (logits)     │    │     772→256→128→64→4           │
│                  │    │         (Sigmoid)              │
└───────┬──────────┘    └──────────────┬────────────────┘
        │                              │
        │ softmax → task_probs         │ per-sample α [B, K]
        │     (detach)                 │
        │         ┌────────────────────┘
        │         │ Batch Mean
        ▼         ▼
   task_logits   α [K]
    [B, K]     合并系数
```

### 3.2 各组件详细说明

#### 3.2.1 共享 Embedding 编码器

- **来源**：GPT-Neo 125M 的 `transformer.wte`（词嵌入层）
- **参数**：50257 × 768（约 38.6M 参数，全部**冻结**不训练）
- **处理流程**：
  1. `embeddings = wte(input_ids)` → `[B, L, 768]`
  2. `detach()` 断开梯度回传
  3. 基于 `attention_mask` 的 **Mean Pooling**：
     $$\mathbf{e}_i = \frac{\sum_{t=1}^{L} m_{i,t} \cdot \mathbf{h}_{i,t}}{\sum_{t=1}^{L} m_{i,t}}$$
     其中 $m_{i,t}$ 为 attention mask, $\mathbf{h}_{i,t}$ 为第 $t$ 个 token 的 embedding
  4. 得到 `sample_embeds` → `[B, 768]`

#### 3.2.2 Task Classifier（Head 2 — 任务分类器）

- **功能**：对每条样本预测其属于哪个任务（4分类）
- **结构**：3层 MLP

| 层 | 结构 | 输出维度 |
|----|------|---------|
| 0 | Linear(768, 128) | 128 |
| 1 | LayerNorm(128) | 128 |
| 2 | ReLU | 128 |
| 3 | Dropout(0.1) | 128 |
| 4 | Linear(128, 64) | 64 |
| 5 | ReLU | 64 |
| 6 | Dropout(0.1) | 64 |
| 7 | Linear(64, 4) | 4 (logits) |

- **输出**：`task_logits` → `[B, K]`（原始 logits，不加 Softmax，由 CrossEntropyLoss 内部处理）
- **可训练参数**：768×128 + 128 + 128 + 128×64 + 64 + 64×4 + 4 = **107,332**

#### 3.2.3 Merge Head（Head 1 — 合并权重预测器, Task-Conditioned）

- **功能**：对每条样本预测其对各任务向量的合并权重 α
- **核心设计**：**任务条件化**（Task-Conditioned）
  - 输入 = `[sample_embeds ⊕ softmax(task_logits).detach()]` → `[B, 772]`
  - Task Classifier 的预测概率 `task_probs` 经 **detach** 传入，作为条件信息
  - `detach` 的目的：merge_loss 的梯度不干扰 Task Classifier 的训练
  - 作用机制：code 样本的 `task_probs ≈ [1, 0, 0, 0]` → Merge Head 产生偏重 code 的 α；math 样本的 `task_probs ≈ [0, 0, 0, 1]` → 偏重 math 的 α
- **结构**：4层 MLP

| 层 | 结构 | 输出维度 |
|----|------|---------|
| 0 | Linear(772, 256) | 256 |
| 1 | LayerNorm(256) | 256 |
| 2 | ReLU | 256 |
| 3 | Dropout(0.1) | 256 |
| 4 | Linear(256, 128) | 128 |
| 5 | LayerNorm(128) | 128 |
| 6 | ReLU | 128 |
| 7 | Dropout(0.1) | 128 |
| 8 | Linear(128, 64) | 64 |
| 9 | ReLU | 64 |
| 10 | Dropout(0.1) | 64 |
| 11 | Linear(64, 4) | 4 |
| 12 | Sigmoid | 4 |

- **输出**：`per_sample_alphas` → `[B, K]`（每个 α_k ∈ [0, 1]，各任务权重独立）
- **最终合并系数**：`α = mean(per_sample_alphas, dim=0)` → `[K]`（Batch 内所有样本的 α 取均值）
- **可训练参数**：772×256 + 256 + 256 + 256×128 + 128 + 128 + 128×64 + 64 + 64×4 + 4 = **239,940**

#### 3.2.4 参数统计

| 组件 | 参数数量 | 是否训练 |
|------|---------|---------|
| Embedding (wte) | ~38.6M | ❌ 冻结 |
| Merge Head | 239,940 | ✅ 训练 |
| Task Classifier | 107,332 | ✅ 训练 |
| **总可训练参数** | **347,272** | — |

### 3.3 关键设计决策

1. **Sigmoid 而非 Softmax**：α 之间互相独立，允许多个任务向量同时高权重（例如 code 和 langid 可能共享语言理解能力）
2. **Per-sample α → Batch 均值**：每条样本独立产生 α，Batch 均值自然反映数据集的任务分布
3. **task_probs detach**：保护 Task Classifier 不受 merge_loss 的梯度干扰，两个 Head 各自专注自己的优化目标
4. **Xavier 初始化**：所有可训练层使用 Xavier uniform 初始化

---

## 4. Task Vector 提取与预处理

### 4.1 Task Vector 定义

对于第 $k$ 个任务，Task Vector 定义为微调模型 Backbone 参数与基座模型参数的逐层差值：

$$\tau^{(k)} = \theta_{\text{ft}}^{(k)} - \theta_{\text{pre}}$$

仅涉及 Backbone 部分的参数（GPT-Neo 中以 `transformer.*` 开头的层），不包含分类头。

### 4.2 提取流程

对每个任务 $k = 1, \ldots, K$：
1. 加载微调模型的 `state_dict`
2. 提取带 `base_model.*` 前缀的参数，去前缀后与基座模型参数对齐
3. 逐层计算 $\tau^{(k)}[\text{name}] = \theta_{\text{ft}}^{(k)}[\text{name}] - \theta_{\text{pre}}[\text{name}]$
4. 提取带 `classification_head.*` 前缀的参数作为冻结的分类头权重
5. 将 Task Vector 和分类头分别序列化保存至磁盘

### 4.3 分类头结构

各任务的分类头结构统一（与微调代码一致）：

```
Linear(768, 768) → LayerNorm(768) → ReLU → Dropout(0.2) →
Linear(768, 384) → ReLU → Dropout(0.1) → Linear(384, num_classes)
```

### 4.4 当前实验中的任务配置

| 任务 | 描述 | 分类数 | 训练样本数 | Task Vector 层数 | 稀疏率 |
|------|------|--------|-----------|-----------------|-------|
| code | 编程语言分类 | 1006 | 47,409 | 160 | ~75% |
| langid | 欧洲语言识别 | 6 | 51,287 | 160 | ~75% |
| law | 法律文档分类(SCOTUS) | 13 | 5,000 | 160 | ~75% |
| math | 数学QA主题分类 | 25 | 35,000 | 160 | ~75% |

> 注：微调过程使用了 25% 掩码率的模块化稀疏微调（Mask 0.25），因此 Task Vector 具有约 75% 的稀疏率。

---

## 5. 训练过程

### 5.1 Dirichlet 混合采样

训练的关键在于**模拟部署场景中用户可能提供的各种任务分布**。每个训练 Batch 的任务比例通过 **Dirichlet 分布**随机采样：

$$\mathbf{p} \sim \text{Dir}(\alpha_d \cdot \mathbf{1}_K)$$

其中 $\alpha_d$ 为浓度参数（默认 0.3）：
- $\alpha_d < 1$：偏向极端分布（某一任务占绝大多数），训练 Router 适应单任务或少任务场景
- $\alpha_d = 1$：均匀分布
- $\alpha_d > 1$：偏向各任务均匀混合

根据采样得到的比例 $\mathbf{p}$，分配 batch 中各任务的样本数 $n_k = \text{round}(p_k \cdot B)$，从各任务数据集中随机采样并 shuffle 组成 Batch。

### 5.2 损失函数

总损失由三部分组成：

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{merge}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{task\_cls}} + \lambda_\alpha \cdot \mathcal{L}_{\text{reg}}$$

#### 5.2.1 合并损失 $\mathcal{L}_{\text{merge}}$

使用 Router 输出的**统一 α**（Batch 内所有样本 α 的均值）合并**唯一一个 Backbone**，然后按样本的 ground truth 任务标签路由至对应的冻结分类头，计算交叉熵损失：

$$\theta^* = \theta_{\text{pre}} + \sum_{k=1}^{K} \alpha_k \cdot \tau^{(k)}$$

$$\mathcal{L}_{\text{merge}} = \frac{\sum_{i=1}^{B} \text{CE}\bigl(H^{(t_i)}(f(\mathbf{x}_i; \theta^*)),\, y_i\bigr)}{B}$$

其中 $f(\mathbf{x}_i; \theta^*)$ 表示使用合并参数 $\theta^*$ 通过 `torch.func.functional_call` 进行无状态前向传播得到的隐状态（经 Mean Pooling + LayerNorm 后处理），$t_i$ 为样本 $i$ 的 ground truth 任务，$H^{(t_i)}$ 为对应的冻结分类头。

**关键技术**：`functional_call` 使得整个合并过程可微分——梯度从 $\mathcal{L}_{\text{merge}}$ 经合并参数 $\theta^*$ 回传到 Router 的 $\alpha$。

**样本数加权**：使用 `CrossEntropyLoss(reduction='sum')` 后除以总样本数 $B$，使梯度信号真实反映 Batch 的任务分布（多数任务的样本贡献更大的梯度）。

#### 5.2.2 任务分类损失 $\mathcal{L}_{\text{task\_cls}}$

标准交叉熵，监督 Task Classifier 识别每条样本属于哪个任务：

$$\mathcal{L}_{\text{task\_cls}} = \frac{1}{B} \sum_{i=1}^{B} \text{CE}(\text{task\_logits}_i,\, t_i)$$

#### 5.2.3 α 稀疏正则化 $\mathcal{L}_{\text{reg}}$

鼓励 per-sample α 趋向 0 或 1（二值化）：

$$\mathcal{L}_{\text{reg}} = \frac{1}{B \cdot K} \sum_{i=1}^{B} \sum_{k=1}^{K} \alpha_{i,k} \cdot (1 - \alpha_{i,k})$$

当 $\alpha_{i,k} \in \{0, 1\}$ 时此项为 0，当 $\alpha_{i,k} = 0.5$ 时最大。

### 5.3 训练配置

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| batch_size | 32 | 每 Batch 的总样本数 |
| num_epochs | 5 | 训练轮数 |
| batches_per_epoch | 1000 | 每轮的 Batch 数 |
| learning_rate | 1e-3 | 初始学习率 |
| weight_decay | 1e-4 | AdamW 权重衰减 |
| $\lambda_{\text{cls}}$ | 0.5 | 任务分类损失权重 |
| $\lambda_\alpha$ | 1.0 | α 正则化权重 |
| $\alpha_d$ (Dirichlet) | 0.3 | Dirichlet 浓度参数 |
| grad_clip_norm | 1.0 | 梯度裁剪范数 |
| LR scheduler | CosineAnnealing | 余弦退火至 lr × 0.01 |
| val_every | 200 steps | 验证频率 |

### 5.4 验证策略

- **固定验证分布**：验证时使用 `dirichlet_alpha=1000`（近似均匀分布）+ 固定种子，确保每次验证条件一致可比较
- **模型选择指标**：**平均归一化 ACC**（各任务 ACC 除以微调基线 ACC 的均值），而非 val_loss
- **每任务验证样本**：500 个
- **验证 Batch**：50 个

### 5.5 冻结与可训练组件

| 组件 | 状态 | 说明 |
|------|------|------|
| 基座模型 $\theta_{\text{pre}}$ | 冻结 | 仅提供计算图结构 |
| Task Vectors $\tau^{(k)}$ | 冻结 | 预计算离线存储 |
| 分类头 $H^{(k)}$ | 冻结 | 来自微调模型 |
| Embedding (wte) | 冻结 | Router 的编码器部分 |
| Merge Head | **可训练** | 239,940 参数 |
| Task Classifier | **可训练** | 107,332 参数 |

---

## 6. 评测过程

评测包含三个维度，逐步从隔离测试到系统级测试：

### 6.1 评估一：合并效果评估（单任务 ACC）

**目的**：隔离测试 Merge Head 的质量——给定纯单任务数据，Router 能否生成合适的 α？

**流程**：
1. 依次加载各任务的**全量测试集**（如 code 23703 个样本）
2. 以 batch 为单位输入 Router，获取 α
3. 使用 α 合并主干，通过 `functional_call` 前向推理
4. 使用 **ground truth** 选择分类头计算 ACC
5. 与微调基线对比，计算归一化 ACC = ACC / Baseline ACC

**指标**：
- 各任务 ACC 及归一化 ACC
- α 值分布（期望单任务输入时对应的 α_k 接近 1）

### 6.2 评估二：任务识别性能评估（混合数据集）

**目的**：隔离测试 Task Classifier 的质量——在不同混合比例下分类器能否正确识别任务？

**混合配置**（固定样本数）：

| 配置 | code | langid | law | math | 分布特征 |
|------|------|--------|-----|------|---------|
| Mix-A | 1000 | 1000 | 1000 | 1000 | 均匀分布 |
| Mix-B | 2000 | 400 | 1200 | 400 | 代码为主 |
| Mix-C | 400 | 400 | 400 | 2800 | 数学为主 |
| Mix-D | 200 | 2400 | 200 | 1200 | 语言识别为主 |

**指标**：
- 整体 Task Classification ACC
- Per-task Recall（各任务召回率）
- Router 输出的 α 值

### 6.3 评估三：端到端系统评估

**目的**：模拟真实部署场景，测试完整系统的端到端性能。

**与评估一的关键区别**：
- 评估一使用 ground truth 选头 → 隔离测试 Merge Head
- 评估三使用 **Task Classifier 自动选头** → 测试完整系统

**流程**：
1. 构造混合数据集（包括 Mix-Full 全量测试 + Mix-A/B/C/D）
2. 所有样本经过 Router，收集 `per_sample_alphas` 和 `task_preds`
3. 按 **Task Classifier 的预测结果**分组（而非 ground truth）
4. 每组用组内 per-sample α 的均值合并独立的 Backbone
5. 每组内所有样本使用该组的合并 Backbone + 对应分类头推理
6. 按 **ground truth 任务** 统计各任务 ACC

**指标**：
- 各任务 ACC 及归一化 ACC
- **错头率**（Task Classifier 误判导致使用错误分类头的比例）
- 各组的 α 分布

---

## 7. 算法伪代码

### 算法 1：基于双头 Router 的任务向量动态合并算法

```
输入: 预训练模型参数 θ_pre, K 个微调模型 {θ_ft^(k)}, K 个分类头 {H^(k)},
      训练数据集 D_train, 验证数据集 D_val
输出: 训练好的 Router R_φ

========================================
  阶段一: Task Vector 提取 (离线, 一次性)
========================================
1:  for k = 1 to K do
2:      τ^(k) ← θ_ft^(k) - θ_pre                   // 逐层计算 task vector
3:      H^(k) ← extract_head(θ_ft^(k))              // 提取分类头权重
4:      将 τ^(k), H^(k) 序列化保存至磁盘
5:  end for
6:  冻结 θ_pre, {τ^(k)}, {H^(k)}                     // 全部不参与训练

========================================
  阶段二: Router 训练
========================================
7:  初始化双头 Router R_φ = (MergeHead_φ1, TaskClassifier_φ2)
8:  // MergeHead: Linear(772→256)→LN→ReLU→Linear(256→128)→LN→ReLU→Linear(128→64)→ReLU→Linear(64→K)→Sigmoid
9:  // TaskClassifier: Linear(768→128)→LN→ReLU→Linear(128→64)→ReLU→Linear(64→K)
10: 设 Encoder = 冻结的 θ_pre.transformer.wte          // 共享 Embedding
11: 初始化 AdamW 优化器, CosineAnnealing 学习率调度器
12:
13: for epoch = 1 to E do
14:     for step = 1 to S do
15:
16:         // --- 采样 ---
17:         p ~ Dir(α_d · 1_K)                          // Dirichlet 随机任务比例
18:         n_k ← round(p_k · B), Σn_k = B              // 各任务样本数
19:         batch ← 从各任务数据集采样 n_k 个样本, shuffle
20:         (x, y, t) ← batch                            // x:输入, y:标签, t:任务ID
21:
22:         // --- Router Forward ---
23:         e ← MeanPool(Encoder(x).detach(), mask)       // [B, 768] 样本嵌入
24:         task_logits ← TaskClassifier(e)               // [B, K]
25:         task_probs ← softmax(task_logits).detach()    // [B, K] 任务概率(断梯度)
26:         merge_input ← concat(e, task_probs)           // [B, 772]
27:         per_sample_α ← MergeHead(merge_input)         // [B, K] 各样本独立 α
28:         α ← mean(per_sample_α, dim=0)                 // [K] Batch 均值 → 统一合并系数
29:
30:         // --- 动态合并 (可微分) ---
31:         θ* ← θ_pre + Σ_{k=1}^{K} α_k · τ^(k)        // 加权合并(保留计算图)
32:
33:         // --- Merge Loss (使用 functional_call) ---
34:         for each task group g in batch do               // 按 ground truth 任务拆分
35:             h_g ← functional_call(θ_pre, θ*, x_g)     // 无状态前向 → 隐状态
36:             s_g ← LayerNorm(MeanPool(h_g))             // 后处理
37:             ŷ_g ← H^(g)(s_g)                           // 对应冻结分类头
38:         end for
39:         L_merge ← Σ CE(ŷ_g, y_g) / B                  // 样本数加权平均
40:
41:         // --- 辅助损失 ---
42:         L_cls ← CE(task_logits, t) / B                 // 任务分类 CE
43:         L_reg ← mean(per_sample_α ⊙ (1 - per_sample_α))  // α 稀疏正则
44:
45:         // --- 总损失 + 反向传播 ---
46:         L_total ← L_merge + λ_cls · L_cls + λ_α · L_reg
47:         φ ← φ - η · clip(∇_φ L_total, max_norm=1.0)
48:         更新学习率 (CosineAnnealing)
49:
50:     end for
51:
52:     // --- 验证 (每 val_every 步) ---
53:     使用固定均匀分布(Dir(1000·1_K)) + 固定种子采样验证 Batch
54:     计算各任务 ACC, 归一化 ACC = ACC / Baseline_ACC
55:     if 平均归一化 ACC > 历史最佳 then 保存 Router checkpoint
56:
57: end for
```

### 算法 2：推理部署

```
输入: 训练好的 Router R_φ*, 预训练参数 θ_pre, Task Vectors {τ^(k)},
      分类头 {H^(k)}, 用户数据集 D_user
输出: 每条样本的分类结果

========================================
  阶段一: 合并主干网络 (Dataset-level, 一次性)
========================================
1:  将整个 D_user 分 batch 输入 Router
2:  收集所有样本的 per_sample_α, task_preds
3:  按 task_preds 将样本分为 G 个组
4:  for each group g do
5:      α_g ← mean(per_sample_α[group_g])         // 组内 α 均值
6:      θ*_g ← θ_pre + Σ_{k=1}^{K} α_g_k · τ^(k) // 该组的合并 backbone
7:  end for

========================================
  阶段二: 分类推理 (Sample-level)
========================================
8:  for each group g do
9:      for each batch in group_g do
10:         h ← forward(x; θ*_g)                   // 合并 backbone 推理
11:         s ← LayerNorm(MeanPool(h))
12:         ŷ ← H^(pred_task_g)(s)                 // Task Classifier 预测的头
13:         results ← argmax(ŷ)
14:     end for
15: end for
16: return results
```

---

## 8. 梯度流分析

理解系统各部分的梯度流是复现的关键：

```
L_merge ──→ CE Loss
             │
             ▼
        Classification Head (冻结, 梯度穿过)
             │
             ▼
        LayerNorm + MeanPool (梯度穿过)
             │
             ▼
        functional_call(θ*) ── θ* = θ_pre + Σ α_k · τ^(k)
                                                │
                                   ┌────────────┘
                                   ▼
                                  α_k (Router Merge Head 的输出)
                                   │
                              ┌────┘
                              ▼
                         MergeHead 参数 φ1 ← 更新
                              ▲
                              │
                    ┌─────────┘ (merge_input = [embed ⊕ task_probs.detach()])
                    │
                    │  embed 来自冻结 Embedding (不更新 wte)
                    │  task_probs 经 detach (不影响 TaskClassifier)
                    │
L_cls ───→ CE Loss → TaskClassifier 参数 φ2 ← 独立更新

L_reg ───→ per_sample_α → MergeHead 参数 φ1 ← 同时更新
```

**要点**：
- $\mathcal{L}_{\text{merge}}$ 只更新 Merge Head（经 detach 隔离了 Task Classifier）
- $\mathcal{L}_{\text{cls}}$ 只更新 Task Classifier
- $\mathcal{L}_{\text{reg}}$ 只更新 Merge Head
- 基座模型、Task Vector、分类头、Embedding 全部冻结

---

## 9. 复现要点

### 9.1 环境依赖

- Python 3.8+
- PyTorch 2.0+（需支持 `torch.func.functional_call`）
- Transformers（HuggingFace）
- GPT-Neo 125M 预训练权重
- 4 个任务的微调模型权重（使用 25% 掩码率的模块化稀疏微调）

### 9.2 数据处理

- **Tokenizer**：GPT2Tokenizer（pad_token = eos_token）
- **Max Length**：512 tokens
- **Padding**：`max_length`（固定长度填充）
- **标签映射**：字符串标签使用 `sorted(unique_labels)` → 索引映射（保证与微调代码一致）

### 9.3 隐状态后处理

合并模型输出的隐状态后处理必须与微调代码一致：

```python
# 与微调中 GPTNeoWithClassificationHead.forward 完全一致
sentence_repr = torch.mean(last_hidden_state, dim=1)   # Mean Pooling
ln = nn.LayerNorm(hidden_size).to(device)               # 默认参数 (weight=1, bias=0)
sentence_repr = ln(sentence_repr)                       # LayerNorm
```

> 注意：微调代码中 LayerNorm 在 forward 中每次新建（使用默认参数），因此这里也使用未训练的 LayerNorm。

### 9.4 functional_call 用法

```python
from torch.func import functional_call

output = functional_call(
    base_model,        # GPTNeoForCausalLM 实例（仅提供计算图结构）
    merged_params,     # 合并后的参数字典（保留 α 的梯度）
    args=(),
    kwargs={'input_ids': input_ids, 'attention_mask': attention_mask}
)
hidden_states = output.hidden_states[-1]  # 最后一层隐状态
```

需设置 `base_model.config.output_hidden_states = True`。

### 9.5 执行步骤

```bash
# 1. 提取 Task Vectors 和分类头（一次性）
python router/task_vectors.py

# 2. 训练 Router
python router/train.py \
  --num_epochs 5 \
  --batches_per_epoch 1000 \
  --batch_size 32 \
  --lr 0.001 \
  --lambda_task_cls 0.5 \
  --dirichlet_alpha 0.3 \
  --lambda_alpha 1.0 \
  --val_every 200

# 3. 评估
python router/evaluate.py \
  --checkpoint router/checkpoints/<run_dir>/best_router.pt
```

---

## 10. 实验结果参考

### 10.1 单任务合并效果

| 任务 | 合并 ACC | 微调基线 | 归一化 |
|------|---------|---------|--------|
| code | 82.97% | 83.90% | 98.89% |
| langid | 87.00% | 91.73% | 94.84% |
| law | 64.50% | 70.57% | 91.40% |
| math | 89.21% | 95.85% | 93.07% |
| **平均** | — | — | **94.55%** |

### 10.2 任务分类器性能

| 混合配置 | Task ACC |
|---------|---------|
| Mix-A (均匀) | 96.88% |
| Mix-B (代码主导) | 96.75% |
| Mix-C (数学主导) | 96.62% |
| Mix-D (语言主导) | 98.25% |

### 10.3 端到端性能

| 混合配置 | code | langid | law | math | 平均归一化 |
|---------|------|--------|-----|------|----------|
| Mix-Full | 81.37% | 86.92% | 64.37% | 88.80% | 93.90% |
| Mix-A | 79.60% | 87.20% | 63.90% | 89.20% | 93.39% |
| Mix-B | 81.80% | 91.47% | 62.88% | 89.87% | 95.02% |
| Mix-C | 83.20% | 84.80% | 64.80% | 88.87% | 94.04% |
| Mix-D | 77.20% | 87.20% | 63.60% | 89.16% | 92.55% |
