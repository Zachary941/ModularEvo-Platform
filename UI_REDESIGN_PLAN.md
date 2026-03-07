# 🎨 LLMOps Platform — UI 美化方案

> **目标**: 在保留全部核心算法逻辑和 API 调用的前提下，将界面从"功能走通"级别提升到"学术论文 Demo 展示"级别，体现**专业、严谨、前沿、轻量**的极简科技风。

---

## 一、现状分析

### 1.1 技术栈（不变）
| 类别 | 技术 |
|------|------|
| 框架 | Vue 3 (Composition API + `<script setup>`) |
| UI 组件库 | Element Plus |
| 图表 | ECharts 6 + vue-echarts |
| 路由 | Vue Router 4 |
| HTTP | Axios |
| 构建 | Vite |
| CSS | 纯自定义 CSS (CSS Custom Properties) |

### 1.2 现有文件 (共 ~3500 行)
| 文件 | 行数 | 角色 |
|------|------|------|
| `style.css` | 108 | 全局变量 & 覆盖 |
| `App.vue` | 32 | 根布局容器 |
| `AppHeader.vue` | 152 | 固定顶栏 |
| `AppSidebar.vue` | 328 | 固定侧边栏 (GPU/系统监控) |
| `HomeView.vue` | 368 | 首页 Hero + 入口卡片 |
| `ModularEvoView.vue` | 839 | 核心页面 (6 张卡片) |
| `AutoRouterView.vue` | 738 | 核心页面 (4 张卡片) |
| `KnowledgeGraph.vue` | 136 | ECharts 力导向图 |
| `TerminalLog.vue` | 136 | 终端日志模拟 |
| `Chapter3View.vue` | 463 | ❌ 旧版，未注册路由 |
| `Chapter4View.vue` | 74 | ❌ 旧版占位，未使用 |
| `HelloWorld.vue` | 43 | ❌ 脚手架默认，未使用 |

### 1.3 现有设计问题诊断

| # | 问题分类 | 具体问题 |
|---|---------|---------|
| 1 | **色彩体系** | 紫色主题偏暗偏沉，缺少科技前沿感；功能色使用不统一 |
| 2 | **字体** | 无等宽字体引入（参数/代码区不够专业）；缺少 Inter/Roboto 等现代无衬线字体 |
| 3 | **排版密度** | 卡片内部间距偏紧，内容紧凑；文字层级区分不够明显 |
| 4 | **视觉层次** | 所有卡片同一层级，缺少视觉锚点和信息分组 |
| 5 | **动效** | Hero 区有装饰圆形但无微动效；按钮交互反馈单一（仅 translateY） |
| 6 | **组件复用** | ModularEvoView 和 AutoRouterView 有大量重复的样式代码（模型信息、算法步骤等） |
| 7 | **数据可视化** | ECharts 使用原始配置，缺乏统一的科技感主题调色 |
| 8 | **深色终端** | TerminalLog 样式尚可但与整体 Light Mode 不太融合 |
| 9 | **侧边栏** | Emoji 作为图标不够精致；进度条与整体风格不够协调 |
| 10 | **首页** | Hero 区内容偏平，缺少动态元素；算法卡片图片与文字比例失衡 |

---

## 二、设计规范 (Design Tokens)

### 2.1 色彩体系 — "量子蓝 + 霓虹青" 双色调

```
主色（量子蓝）:
  --c-primary-50:  #eef2ff    背景淡色
  --c-primary-100: #e0e7ff    卡片高亮底色
  --c-primary-200: #c7d2fe    边框/分割线高亮
  --c-primary-400: #818cf8    次要按钮/图标
  --c-primary-500: #6366f1    主按钮/主标题强调     ← 主色
  --c-primary-600: #4f46e5    按钮悬停
  --c-primary-700: #4338ca    深色文字强调
  --c-primary-900: #1e1b4b    标题文字

强调色（霓虹青）:
  --c-accent-400:  #2dd4bf
  --c-accent-500:  #14b8a6    成功/高亮/进度      ← 辅助强调色

功能色:
  --c-success:     #10b981    成功
  --c-warning:     #f59e0b    警告
  --c-danger:      #ef4444    错误
  --c-info:        #6366f1    信息（与主色统一）

中性色:
  --c-gray-50:     #f9fafb    页面背景
  --c-gray-100:    #f3f4f6    卡片组背景
  --c-gray-200:    #e5e7eb    边框
  --c-gray-400:    #9ca3af    次要文字
  --c-gray-500:    #6b7280    正文
  --c-gray-700:    #374151    标题
  --c-gray-900:    #111827    终端背景/最深色
```

### 2.2 字体栈

```css
/* 正文 — 清晰现代无衬线 */
--font-sans: 'Inter', 'PingFang SC', 'Noto Sans SC', -apple-system, sans-serif;

/* 代码 / 参数 / 数值 — 等宽字体 */
--font-mono: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'SF Mono', monospace;
```

> 通过 Google Fonts CDN 在 `index.html` 中引入 Inter + JetBrains Mono。

### 2.3 间距系统

```
--space-xs:  4px
--space-sm:  8px
--space-md:  16px
--space-lg:  24px
--space-xl:  32px
--space-2xl: 48px
```

### 2.4 圆角

```
--radius-sm:  6px    按钮内元素
--radius-md:  10px   卡片内区块
--radius-lg:  14px   卡片
--radius-xl:  20px   Hero区/大模块
```

### 2.5 阴影

```
--shadow-sm:   0 1px 2px rgba(0,0,0,0.04);
--shadow-md:   0 4px 12px rgba(99,102,241,0.06);
--shadow-lg:   0 8px 30px rgba(99,102,241,0.10);
--shadow-glow: 0 0 20px rgba(99,102,241,0.15);   /* 主色发光 */
```

### 2.6 动效约定

```
--ease-out:   cubic-bezier(0.16, 1, 0.3, 1);   /* 自然缓出 */
--ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1); /* 弹性 */
--duration-fast:   150ms;
--duration-normal: 250ms;
--duration-slow:   400ms;
```

---

## 三、美化方案详述

### 3.1 全局样式 (`style.css`)

**改动要点**:
- 替换所有 CSS 变量为新的设计 Token
- 引入字体变量 `--font-sans` 和 `--font-mono`
- 增强 `el-card` 覆盖：轻微毛玻璃背景 (`backdrop-filter`)，悬停发光阴影
- `el-button--primary` 使用量子蓝渐变，悬停时微发光
- 统一 `el-tag` 样式：圆润胶囊型 + 半透明底色
- 增加平滑过渡工具类 (`.fade-in`, `.slide-up`)

### 3.2 `index.html` — 字体引入

在 `<head>` 中通过 Google Fonts CDN 引入 `Inter` (400/500/600/700/800) 和 `JetBrains Mono` (400/500/600)。

### 3.3 `AppHeader.vue` — 顶部导航栏

**改动要点**:
- 背景从纯紫色渐变改为**深灰蓝 + 量子蓝微光**渐变 (`#0f172a → #1e1b4b`)，更加沉稳专业
- Logo 区域增加一个微小的圆形光晕动画（呼吸灯效果）
- 导航菜单项悬停时底部出现细线指示器（2px 宽 accent 色），而非背景变色
- 右侧状态指示器加入 `backdrop-filter: blur(8px)` 毛玻璃背景
- 使用 CSS `transition` 让菜单项切换更丝滑

### 3.4 `AppSidebar.vue` — 侧边栏

**改动要点**:
- 背景改为纯白 + 极浅灰底纹 (`#f9fafb`)，去掉紫色渐变
- 各 Section 标题前的 emoji 替换为 **SVG 内联图标或 Element Plus Icons**，更专业
- GPU/内存进度条使用自定义的细线形进度条（扁平化设计），颜色使用设计 Token
- 卡片区块增加 `border-left: 3px solid var(--c-primary-200)` 作为视觉分组
- 数值文字使用 `--font-mono`（等宽字体），对齐更整洁
- 运行任务状态使用小圆点 + 颜色（绿色=活跃、灰色=空闲）替代 emoji
- 整体间距调大，呼吸感更强

### 3.5 `HomeView.vue` — 首页

**改动要点**:
- **Hero 区**:
  - 背景改为微妙的网格点阵图案 + 径向渐变光晕（CSS `radial-gradient` 模拟光斑）
  - 装饰圆形增加缓慢浮动动画（`@keyframes float`，上下 8px，周期 6s）
  - 标题渐变色使用量子蓝→霓虹青
  - 统计数字使用 `--font-mono` 等宽字体，视觉更锐利
  - Badge 区域加入微光流动边框（`border-image` 渐变动画）

- **算法入口卡片**:
  - 图片侧面光晕从渐变 div 改为 `box-shadow` + `filter: blur()` 实现更自然的外发光
  - 卡片悬停时图片微放大 + 整体向上浮起 + 阴影加深的组合动效
  - Tag 使用半透明底色 + 小圆角胶囊型
  - "进入演示" 按钮改为线框样式 + 悬停时填充（更克制）

### 3.6 `ModularEvoView.vue` — 核心页面

**改动要点**:
- **算法步骤卡片 (Step 1/2/3)**:
  - 三个步骤块使用渐变色顶部边线（3px 宽），分别使用不同色调（蓝/绿/紫）
  - 步骤之间的箭头 `→` 替换为 SVG 箭头图标 + 连接线，更正式
  - 每个步骤块悬停时微上浮 + 阴影扩散

- **知识图谱卡片**:
  - 卡片内部增加深色背景变体（图谱区域使用 `#fafbff` 极浅蓝底色）
  - 图谱容器加入细边框，解决图谱与卡片白色背景融合的问题

- **模型信息卡片**:
  - 左侧树保持不变
  - 右侧模型详情卡片字段值使用 `--font-mono` 等宽字体
  - 颜色方案保持但调淡，让文字更突出

- **模块化卡片**:
  - 模块行使用卡片式布局（微底色 + 圆角容器），替代简单的 `border-bottom` 分割
  - ECharts 柱状图应用统一主题色 + 圆角柱体 + 阴影
  - "已加载" 状态使用发光的绿色指示器 + 渐入动画

- **稀疏微调卡片**:
  - 两列卡片使用渐变顶部色条区分任务
  - 评测指标表格使用斑马纹 + 高亮关键数值（主色加粗）
  - 按钮状态变化使用平滑色彩过渡

- **模型合并卡片**:
  - 参数配置区域使用毛玻璃效果面板
  - 合并方法选择器增加自定义图标前缀
  - 终端日志与图表的布局使用 CSS Grid 弹性切换
  - 对比柱状图增加网格线 + 更柔和的配色方案

### 3.7 `AutoRouterView.vue` — 第二核心页面

**改动要点**（与 ModularEvoView 对称）:
- 算法步骤和知识图谱样式与 ModularEvo 保持一致
- **数据集上传区域**:
  - 使用虚线边框 + 拖拽区域样式（dashed border + hover 高亮）
  - 文件名/大小显示使用 `--font-mono`
  - 上传预览区的 Tag 排列更紧凑

- **评测结果面板**:
  - α 系数柱状图使用更丰富的配色 + 高亮最大值
  - 归一化性能图的 Baseline 线使用虚线 + 标注更醒目
  - 汇总表格的关键数值使用大号加粗 + 主色高亮

### 3.8 `KnowledgeGraph.vue` — 知识图谱组件

**改动要点**:
- 节点颜色统一到新设计 Token
- 增加选中节点环形光晕效果 (`emphasis.itemStyle.shadowBlur`)
- 边的箭头使用半透明渐变色
- 交互增强：悬停节点时相关边高亮，其余淡化
- 容器背景使用极浅蓝底色 + 细边框

### 3.9 `TerminalLog.vue` — 终端日志组件

**改动要点**:
- 保持深色终端风格（这是好的设计决策）
- 标题栏与终端主体之间增加一条细微的渐变分割线
- 日志文字使用 `--font-mono` 等宽字体（统一字体变量）
- 增加日志出现时的逐行淡入动效（`@keyframes fade-slide-in`，高度 & 透明度）
- 光标闪烁动画从硬切换改为正弦波式柔和闪烁

### 3.10 公共组件抽取

为减少 ModularEvoView.vue 和 AutoRouterView.vue 中的重复代码，抽取以下组件：

| 新组件 | 职责 | 复用位置 |
|--------|------|---------|
| `AlgoSteps.vue` | 算法三步流程展示 | ModularEvoView, AutoRouterView |
| `ModelInfoPanel.vue` | 模型家族树 + 详情卡片 | ModularEvoView, AutoRouterView |
| `SectionCard.vue` | 统一的带图标标题卡片容器 | 所有页面 |

### 3.11 ECharts 统一主题

创建 `src/utils/chartTheme.js`，定义统一的 ECharts 主题配置:
- 柱状图圆角: `borderRadius: [4, 4, 0, 0]`
- 调色盘使用量子蓝系列
- Tooltip 使用毛玻璃背景
- 坐标轴线淡化、网格线虚线化
- 字体统一为 `--font-mono`

---

## 四、文件变更清单

| 操作 | 文件 | 说明 |
|------|------|------|
| ✏️ 修改 | `index.html` | 添加 Google Fonts CDN 链接 |
| ✏️ 修改 | `src/style.css` | 重写设计 Token + 全局覆盖样式 |
| ✏️ 修改 | `src/App.vue` | 微调布局变量 |
| ✏️ 修改 | `src/components/AppHeader.vue` | 重写样式，保留逻辑 |
| ✏️ 修改 | `src/components/AppSidebar.vue` | 重写样式，保留逻辑和数据流 |
| ✏️ 修改 | `src/views/HomeView.vue` | 重写 Hero + 卡片样式 |
| ✏️ 修改 | `src/views/ModularEvoView.vue` | 重写 6 张卡片样式，保留全部 API 逻辑 |
| ✏️ 修改 | `src/views/AutoRouterView.vue` | 重写 4 张卡片样式，保留全部上传/评测逻辑 |
| ✏️ 修改 | `src/components/KnowledgeGraph.vue` | 更新节点配色和交互效果 |
| ✏️ 修改 | `src/components/TerminalLog.vue` | 微调字体和动效 |
| ➕ 新增 | `src/components/AlgoSteps.vue` | 抽取算法步骤展示组件 |
| ➕ 新增 | `src/components/ModelInfoPanel.vue` | 抽取模型信息面板组件 |
| ➕ 新增 | `src/components/SectionCard.vue` | 抽取统一卡片容器组件 |
| ➕ 新增 | `src/utils/chartTheme.js` | ECharts 统一主题配置 |
| 🗑️ 可选删除 | `src/views/Chapter3View.vue` | 未注册路由的旧版页面 |
| 🗑️ 可选删除 | `src/views/Chapter4View.vue` | 未注册路由的占位页面 |
| 🗑️ 可选删除 | `src/components/HelloWorld.vue` | Vite 默认组件 |

---

## 五、执行步骤

### Phase 1: 设计基础设施 (Design Foundation)
1. **Step 1**: 修改 `index.html`，引入 Inter + JetBrains Mono 字体
2. **Step 2**: 重写 `src/style.css`，建立新的设计 Token 系统
3. **Step 3**: 创建 `src/utils/chartTheme.js`，定义 ECharts 统一主题

### Phase 2: 布局骨架 (Layout Shell)
4. **Step 4**: 修改 `App.vue`，适配新的布局变量
5. **Step 5**: 重写 `AppHeader.vue` 样式
6. **Step 6**: 重写 `AppSidebar.vue` 样式

### Phase 3: 公共组件 (Shared Components)
7. **Step 7**: 创建 `SectionCard.vue`
8. **Step 8**: 创建 `AlgoSteps.vue`
9. **Step 9**: 创建 `ModelInfoPanel.vue`
10. **Step 10**: 更新 `KnowledgeGraph.vue` 配色和交互
11. **Step 11**: 更新 `TerminalLog.vue` 字体和动效

### Phase 4: 页面重写 (Page Restyling)
12. **Step 12**: 重写 `HomeView.vue`
13. **Step 13**: 重写 `ModularEvoView.vue` — 使用新公共组件，保留全部 script 逻辑
14. **Step 14**: 重写 `AutoRouterView.vue` — 使用新公共组件，保留全部 script 逻辑

### Phase 5: 清理与验证 (Cleanup & QA)
15. **Step 15**: 删除未使用的旧文件 (Chapter3View, Chapter4View, HelloWorld)
16. **Step 16**: 运行 `npm run dev` 验证编译无误
17. **Step 17**: 逐页面目视检查 UI 效果，微调间距和配色

---

## 六、注意事项

1. **绝对不动的部分**: 所有 `<script setup>` 中的 API 调用函数、`computed`、`ref`、`onMounted`、事件处理函数、ECharts 数据计算逻辑。
2. **模板层**: HTML 结构可以重构（如用新组件包裹），但所有 `v-for`、`v-if`、事件绑定 (`@click`)、`:option`、`:lines` 等绑定关系必须保持完整。
3. **无 Tailwind**: 严格使用 CSS Custom Properties + scoped styles，不引入额外 CSS 框架。
4. **响应式**: 当前页面无移动端适配需求（学术演示以桌面为主），但预留 `max-width` 和弹性布局。
5. **性能**: 字体通过 `font-display: swap` 防止 FOIT/FOUT 闪烁；ECharts 保持按需导入。

---

## 七、效果预期

| 维度 | 当前 | 目标 |
|------|------|------|
| 色彩 | 深紫偏暗 | 量子蓝 + 霓虹青，清爽专业 |
| 字体 | 系统默认 | Inter + JetBrains Mono，科技感强 |
| 间距 | 偏紧凑 | 充裕呼吸感，信息层级清晰 |
| 动效 | 基本 hover | 丝滑淡入/浮升/发光，克制高级 |
| 组件化 | 重复代码多 | 公共组件抽取，维护性提升 |
| 图表 | 默认主题 | 统一配色 + 圆角柱/毛玻璃 Tooltip |
| 整体观感 | 可用但朴素 | 专业学术 Demo 水准 |

---

## 八、问题确认

在开始实施前，我想确认以下几个问题：

1. **关于深色模式**: 方案中以 Light Mode 为主设计。你的展示场景是否需要提供 Dark Mode 切换？如果需要，我会在设计 Token 中预设 `[data-theme="dark"]` 变量。
回答：预设吧。

2. **关于图片资源**: 首页的两张算法图片 (`modularevo.png`, `router.png`) 是否需要重新设计或替换？还是保持现有图片仅优化展示效果？
回答：保持现有图片仅优化展示效果。

3. **关于旧文件清理**: `Chapter3View.vue`、`Chapter4View.vue`、`HelloWorld.vue` 三个未使用的文件是否可以直接删除？
回答：可以删除

4. **关于性能指标数字**: 侧边栏的 GPU/系统状态数据目前是真实后端轮询的。是否希望在无 GPU 环境下用 Mock 数据保持界面完整性？
回答：可以

5. **关于 ECharts 图表交互**: 是否希望图表增加数据点击事件（如点击柱状图跳转到模型详情）？还是保持当前的纯展示模式？
回答：可以点击后在下面的模型信息卡片中切换到点击的那个模型的信息。

请回复你的选择后，我将按照上述步骤逐步实施美化方案。
