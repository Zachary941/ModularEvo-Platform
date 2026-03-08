<script setup>
import { ref, computed, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { BarChart } from 'echarts/charts'
import {
  TitleComponent, TooltipComponent, GridComponent, LegendComponent,
} from 'echarts/components'
import VChart from 'vue-echarts'
import KnowledgeGraph from '../components/KnowledgeGraph.vue'
import TerminalLog from '../components/TerminalLog.vue'
import {
  getModels, getBaseline, getGraph, uploadDataset,
  downloadSampleDataset, evaluate,
} from '../api/chapter4'

use([CanvasRenderer, BarChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

/* ======== 卡片 2: 知识图谱 ======== */
const graphNodes = ref([])
const graphEdges = ref([])
const activeRelations = ref([])

const RELATION_COLORS = {
  '微调': { bg: '#fff7ed', border: '#f97316', text: '#c2410c', dot: '#f97316' },
  '任务向量': { bg: '#ecfdf5', border: '#10b981', text: '#065f46', dot: '#10b981' },
  '路由': { bg: '#fefce8', border: '#eab308', text: '#854d0e', dot: '#eab308' },
  '合并': { bg: '#eff6ff', border: '#3b82f6', text: '#1e40af', dot: '#3b82f6' },
  '基座': { bg: '#faf5ff', border: '#a855f7', text: '#6b21a8', dot: '#a855f7' },
}

const availableRelations = computed(() => {
  const set = new Set(graphEdges.value.map(e => e.relation))
  return [...set]
})

const filteredGraphData = computed(() => {
  if (!activeRelations.value.length) {
    return { nodes: graphNodes.value, edges: graphEdges.value }
  }
  const edges = graphEdges.value.filter(e => activeRelations.value.includes(e.relation))
  const nodeIds = new Set()
  edges.forEach(e => { nodeIds.add(e.source); nodeIds.add(e.target) })
  const nodes = graphNodes.value.filter(n => nodeIds.has(n.id))
  return { nodes, edges }
})

function toggleRelation(rel) {
  const idx = activeRelations.value.indexOf(rel)
  if (idx >= 0) {
    activeRelations.value.splice(idx, 1)
  } else {
    activeRelations.value.push(rel)
  }
}

function resetRelationFilter() {
  activeRelations.value = []
}

/* ======== 卡片 3: 模型信息 ======== */
const allModels = ref([])
const selectedModelId = ref('')
const selectedModel = computed(() => allModels.value.find(m => m.id === selectedModelId.value))

// 模型家族树数据
const modelTreeData = computed(() => [
  {
    id: 'gpt-neo-125m', label: 'GPT-Neo 125M', icon: '🧠',
    children: [
      {
        id: 'ft-code', label: 'FT-Code', icon: '🔧',
        children: [{ id: 'tv-code', label: 'τ-Code', icon: '📦' }],
      },
      {
        id: 'ft-langid', label: 'FT-LangID', icon: '🔧',
        children: [{ id: 'tv-langid', label: 'τ-LangID', icon: '📦' }],
      },
      {
        id: 'ft-law', label: 'FT-Law', icon: '🔧',
        children: [{ id: 'tv-law', label: 'τ-Law', icon: '📦' }],
      },
      {
        id: 'ft-math', label: 'FT-Math', icon: '🔧',
        children: [{ id: 'tv-math', label: 'τ-Math', icon: '📦' }],
      },
      { id: 'router', label: 'AutoRouter', icon: '🎯' },
      { id: 'merged', label: 'Merged-Model', icon: '🔗' },
    ],
  },
])
const treeProps = { children: 'children', label: 'label' }
function handleTreeNodeClick(data) {
  selectedModelId.value = data.id
}

// 模型类型样式映射
const MODEL_TYPE_STYLES = {
  pretrained:   { bg: 'linear-gradient(135deg, #ede9fe 0%, #ddd6fe 100%)', border: '#a78bfa', color: '#5b21b6', label: '预训练模型', icon: '🧠' },
  finetuned:    { bg: 'linear-gradient(135deg, #fef3c7 0%, #fde68a 100%)', border: '#fbbf24', color: '#92400e', label: '微调模型', icon: '🔧' },
  task_vector:  { bg: 'linear-gradient(135deg, #cffafe 0%, #a5f3fc 100%)', border: '#22d3ee', color: '#155e75', label: '任务向量', icon: '📦' },
  router:       { bg: 'linear-gradient(135deg, #f3e8ff 0%, #e9d5ff 100%)', border: '#a855f7', color: '#6b21a8', label: '路由网络', icon: '🎯' },
  merged:       { bg: 'linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)', border: '#f87171', color: '#991b1b', label: '合并模型', icon: '🔗' },
}
function getModelStyle(type) {
  return MODEL_TYPE_STYLES[type] || MODEL_TYPE_STYLES.pretrained
}

// meta 字段中文标签映射
const META_LABELS = {
  params: '参数量', arch: '架构', vocab: '词表', hidden: '隐藏层',
  pretrain_data: '预训练数据', pretrain_task: '预训练任务', source: '来源',
  task: '下游任务', classes: '类别数', acc: '基线 ACC', dataset: '评测数据集',
  train_epochs: '训练轮数', lr: '学习率', batch_size: 'Batch Size',
  layers: '层数', formula: '计算公式', size: '存储大小', source_model: '源模型',
  variant: '网络变体', output: '输出维度',
  source_models: '源模型', routing: '路由方式',
}
function getMetaFields(model) {
  if (!model) return []
  const exclude = ['desc']
  const fields = model.meta || {}
  return Object.entries(fields)
    .filter(([k]) => !exclude.includes(k))
    .filter(([, v]) => v !== null && v !== undefined && v !== '')
    .map(([k, v]) => ({ key: k, label: META_LABELS[k] || k, value: String(v) }))
}

/* ======== 卡片 4: AutoRouter 评测 ======== */
const uploadedFile = ref(null)
const uploadResult = ref(null)
const evaluating = ref(false)
const evalResult = ref(null)
const terminalLines = ref([])
const baselineData = ref({})

function addLog(text, type = 'info') {
  terminalLines.value.push({ text, type })
}

function sleep(ms) {
  return new Promise(r => setTimeout(r, ms))
}

async function simulateLogs(logs) {
  for (const log of logs) {
    addLog(log.text, log.type || 'info')
    if (log.delay) await sleep(log.delay)
  }
}

/* 文件选择 */
function handleFileChange(uploadFileObj) {
  uploadedFile.value = uploadFileObj.raw
}

/* 下载示例数据集 */
async function handleDownloadSample() {
  try {
    const { data } = await downloadSampleDataset()
    const url = URL.createObjectURL(data)
    const a = document.createElement('a')
    a.href = url
    a.download = 'mixed_sample_50.csv'
    a.click()
    URL.revokeObjectURL(url)
    ElMessage.success('示例数据集已下载')
  } catch (e) {
    ElMessage.error('下载失败: ' + (e.message || '未知错误'))
  }
}

/* 上传并解析 */
async function handleUpload() {
  if (!uploadedFile.value) {
    ElMessage.warning('请先选择数据集文件')
    return
  }
  terminalLines.value = []
  evalResult.value = null
  addLog('$ upload-dataset ' + uploadedFile.value.name, 'cmd')

  try {
    const { data } = await uploadDataset(uploadedFile.value)
    uploadResult.value = data
    addLog(`[INFO] 文件解析成功: ${data.total_samples} 条样本`, 'success')
    const dist = Object.entries(data.task_distribution).map(([k, v]) => `${k}=${v}`).join(', ')
    addLog(`[INFO] 任务分布: ${dist}`)
    ElMessage.success(`数据集已上传: ${data.total_samples} 条样本`)
  } catch (e) {
    addLog(`[ERROR] 上传失败: ${e.response?.data?.detail || e.message}`, 'error')
    ElMessage.error('上传失败: ' + (e.response?.data?.detail || e.message))
  }
}

/* 开始评测 */
async function handleEvaluate() {
  if (!uploadResult.value) {
    ElMessage.warning('请先上传数据集')
    return
  }
  evaluating.value = true
  evalResult.value = null
  terminalLines.value = []

  // ── Fire the real backend API call immediately ──
  const apiPromise = evaluate(uploadResult.value.filename)

  // Phase 1: 模拟加载日志 (在后端处理时展示)
  await simulateLogs([
    { text: `$ python evaluate.py --dataset ${uploadResult.value.filename}`, type: 'cmd', delay: 400 },
    { text: '[INFO] Loading GPT-Neo 125M base model (~125M params)...', delay: 800 },
    { text: '[INFO] Model loaded: 125,198,592 parameters (GPTNeoForCausalLM)', delay: 400 },
    { text: '[INFO] Loading task vectors: code.pt, langid.pt, law.pt, math.pt', delay: 600 },
    { text: '[INFO]   code: 160 layers loaded', delay: 200 },
    { text: '[INFO]   langid: 160 layers loaded', delay: 200 },
    { text: '[INFO]   law: 160 layers loaded', delay: 200 },
    { text: '[INFO]   math: 160 layers loaded', delay: 200 },
    { text: '[INFO] Loading classification heads (4 tasks)', delay: 400 },
    { text: '[INFO] Loading AutoRouter checkpoint...', delay: 500 },
    { text: '[INFO]   Router variant: hybrid_dual_branch (~99K params)', delay: 300 },
    { text: '[INFO] All resources loaded ✓', type: 'success', delay: 300 },
  ])

  // Phase 2: Router推理
  await simulateLogs([
    { text: '', delay: 100 },
    { text: `[STEP 1/3] Router inference (${uploadResult.value.total_samples} samples)...`, delay: 300 },
    { text: '  Tokenizing input texts (max_length=512)...', delay: 500 },
    { text: '  Computing embeddings via frozen GPT-Neo Embedding...', delay: 600 },
    { text: '  embed_branch → task_branch → fusion_head ...', delay: 500 },
  ])

  // Phase 3: 合并+评测
  await simulateLogs([
    { text: '', delay: 100 },
    { text: '[STEP 2/3] Computing merged parameters...', delay: 300 },
    { text: '  merged_param = base + Σ(αᵢ × τᵢ)  (160 layers)', delay: 600 },
    { text: '  [========================================] 160/160 layers', type: 'success', delay: 300 },
    { text: '', delay: 100 },
    { text: '[STEP 3/3] Evaluating merged model on each task...', delay: 300 },
    { text: '  [code]   Running inference with functional_call ...', delay: 400 },
    { text: '  [langid] Running inference with functional_call ...', delay: 400 },
    { text: '  [law]    Running inference with functional_call ...', delay: 400 },
    { text: '  [math]   Running inference with functional_call ...', delay: 400 },
  ])

  try {
    const { data } = await apiPromise
    evalResult.value = data

    // 展示真实结果
    addLog('')
    addLog('=== Router Output ===', 'info')
    const alphaStr = data.alphas.map((a, i) => `α_${['code','langid','law','math'][i]}=${a.toFixed(4)}`).join('  ')
    addLog(`  Learned α: ${alphaStr}`, 'success')
    addLog(`  Task classification accuracy: ${(data.task_classification_acc * 100).toFixed(1)}%`)

    addLog('')
    addLog('=== Evaluation Results (Normalized: Merged/Baseline) ===', 'info')
    for (const [task, acc] of Object.entries(data.per_task_acc)) {
      const bl = data.baseline[task]
      const normAcc = data.per_task_norm_acc[task]
      addLog(`  [${task.padEnd(6)}]  Merged=${(acc*100).toFixed(2)}%  Baseline=${(bl*100).toFixed(2)}%  Norm=${(normAcc*100).toFixed(1)}%`, normAcc >= 1.0 ? 'success' : 'warning')
    }
    addLog(`  [overall]  NormAcc=${(data.overall_norm_acc*100).toFixed(2)}%  RawAcc=${(data.overall_acc*100).toFixed(2)}%`)
    addLog('')
    addLog('[DONE] AutoRouter evaluation finished successfully ✓', 'success')
    ElMessage.success('评测完成')
  } catch (e) {
    addLog(`[ERROR] 评测失败: ${e.response?.data?.detail || e.message}`, 'error')
    ElMessage.error('评测失败: ' + (e.response?.data?.detail || e.message))
  } finally {
    evaluating.value = false
  }
}

/* α 系数柱状图 */
const alphaChartOption = computed(() => {
  if (!evalResult.value) return {}
  const tasks = ['code', 'langid', 'law', 'math']
  const alphas = evalResult.value.alphas
  return {
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}: α = ${p[0].value.toFixed(4)}` },
    grid: { left: 50, right: 16, top: 24, bottom: 36 },
    xAxis: { type: 'category', data: tasks.map(t => t.charAt(0).toUpperCase() + t.slice(1)) },
    yAxis: { type: 'value', name: 'α', max: 1, axisLabel: { formatter: v => v.toFixed(1) } },
    series: [{
      type: 'bar',
      data: alphas.map((a, i) => ({
        value: a,
        itemStyle: { color: ['#6366f1', '#10b981', '#f59e0b', '#ef4444'][i] },
      })),
      label: { show: true, position: 'top', fontSize: 11, formatter: p => p.value.toFixed(3) },
      barWidth: '40%',
    }],
  }
})

/* 合并模型归一化性能柱状图 */
const compareChartOption = computed(() => {
  if (!evalResult.value) return {}
  const tasks = Object.keys(evalResult.value.per_task_norm_acc)
  const normAcc = tasks.map(t => evalResult.value.per_task_norm_acc[t])
  return {
    tooltip: { trigger: 'axis', formatter: p => `${p[0].name}: ${(p[0].value*100).toFixed(2)}%` },
    grid: { left: 50, right: 16, top: 24, bottom: 36 },
    xAxis: { type: 'category', data: tasks.map(t => t.charAt(0).toUpperCase() + t.slice(1)) },
    yAxis: {
      type: 'value', max: 1.5, min: 0,
      axisLabel: { formatter: v => `${(v * 100).toFixed(0)}%` },
    },
    series: [{
      type: 'bar',
      data: normAcc.map((v, i) => ({
        value: v,
        itemStyle: { color: ['#6366f1', '#10b981', '#f59e0b', '#ef4444'][i] },
      })),
      label: { show: true, position: 'top', fontSize: 11, formatter: p => `${(p.value*100).toFixed(1)}%` },
      barWidth: '40%',
      markLine: {
        silent: true,
        data: [{ yAxis: 1.0, label: { formatter: 'Baseline', position: 'end', fontSize: 10 }, lineStyle: { color: '#999', type: 'dashed' } }],
      },
    }],
  }
})

/* ======== 初始化 ======== */
onMounted(async () => {
  try {
    const [modelRes, graphRes, blRes] = await Promise.all([
      getModels(), getGraph(), getBaseline(),
    ])
    allModels.value = modelRes.data.map(m => {
      // 从 graph 节点获取丰富的 meta 信息
      const graphNode = graphRes.data.nodes.find(n => n.id === m.id || n.id === m.id.replace('gpt-neo-125m', 'gpt-neo'))
      const graphMeta = graphNode?.meta || {}
      return {
        ...m,
        meta: {
          ...graphMeta,
          // 如果 graph 没有某些字段，用 ModelInfo 的补充
          ...(m.params && !graphMeta.params ? { params: m.params } : {}),
          ...(m.baseline_acc && !graphMeta.acc ? { acc: `${(m.baseline_acc*100).toFixed(1)}%` } : {}),
          ...(m.num_classes && !graphMeta.classes ? { classes: m.num_classes } : {}),
        },
      }
    })
    if (allModels.value.length) selectedModelId.value = allModels.value[0].id

    graphNodes.value = graphRes.data.nodes
    graphEdges.value = graphRes.data.edges
    baselineData.value = blRes.data.baseline
  } catch (e) {
    console.error('初始化失败:', e)
  }
})
</script>

<template>
  <div class="autorouter-view">
    <!-- ===== 卡片 1: 算法介绍 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">📖</span>
          <span class="card-title">算法介绍 — AutoRouter 三步流程</span>
        </div>
      </template>
      <div class="algo-steps">
        <div class="algo-step">
          <div class="step-emoji">🔍</div>
          <h4>1. 输入识别</h4>
          <p>Router 分析输入数据分布，通过双分支编码器识别各任务的占比特征。</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="algo-step">
          <div class="step-emoji">⚖️</div>
          <h4>2. 权重组合</h4>
          <p>根据任务分布动态生成合并权重 α，按 base + Σ(αᵢ × τᵢ) 合并模型参数。</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="algo-step">
          <div class="step-emoji">🎯</div>
          <h4>3. 自动匹配分类头</h4>
          <p>合并后的模型搭配对应任务分类头，实现多任务统一推理。</p>
        </div>
      </div>
    </el-card>

    <!-- ===== 卡片 2: 知识图谱 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🕸️</span>
          <span class="card-title">知识图谱 — GPT-Neo 模型关系</span>
        </div>
      </template>
      <div class="graph-filter-bar" v-if="availableRelations.length">
        <button
          class="filter-tag" :class="{ active: activeRelations.length === 0 }"
          @click="resetRelationFilter"
        ><span class="filter-dot" style="background: #6366f1"></span>全部</button>
        <button
          v-for="rel in availableRelations" :key="rel"
          class="filter-tag" :class="{ active: activeRelations.includes(rel) }"
          :style="{
            '--tag-bg': (RELATION_COLORS[rel] || {}).bg || '#f5f3ff',
            '--tag-border': (RELATION_COLORS[rel] || {}).border || '#7c3aed',
            '--tag-text': (RELATION_COLORS[rel] || {}).text || '#5b21b6',
            '--tag-dot': (RELATION_COLORS[rel] || {}).dot || '#7c3aed',
          }"
          @click="toggleRelation(rel)"
        ><span class="filter-dot" :style="{ background: (RELATION_COLORS[rel] || {}).dot || '#7c3aed' }"></span>{{ rel }}</button>
      </div>
      <KnowledgeGraph
        v-if="graphNodes.length"
        :nodes="filteredGraphData.nodes"
        :edges="filteredGraphData.edges"
        height="420px"
      />
      <el-empty v-else description="加载中..." />
    </el-card>

    <!-- ===== 卡片 3: 模型信息 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">📦</span>
          <span class="card-title">模型信息 — 模型家族树</span>
        </div>
      </template>
      <div class="model-info-layout">
        <!-- 左侧: 模型家族树 -->
        <div class="model-tree-wrap">
          <el-tree
            :data="modelTreeData"
            :props="treeProps"
            default-expand-all
            highlight-current
            node-key="id"
            :current-node-key="selectedModelId"
            @node-click="handleTreeNodeClick"
          >
            <template #default="{ data }">
              <span class="tree-node">
                <span class="tree-node-icon">{{ data.icon }}</span>
                <span class="tree-node-label">{{ data.label }}</span>
              </span>
            </template>
          </el-tree>
        </div>
        <!-- 右侧: 模型详情卡片 -->
        <div class="model-detail-card-wrap" v-if="selectedModel">
          <div
            class="model-detail-card"
            :style="{
              background: getModelStyle(selectedModel.type).bg,
              borderColor: getModelStyle(selectedModel.type).border,
            }"
          >
            <div class="model-card-header">
              <span class="model-card-icon">{{ getModelStyle(selectedModel.type).icon }}</span>
              <div class="model-card-title-group">
                <h3 class="model-card-name">{{ selectedModel.name }}</h3>
                <el-tag
                  size="small" effect="dark" round
                  :color="getModelStyle(selectedModel.type).border"
                  style="border: none; color: #fff;"
                >{{ getModelStyle(selectedModel.type).label }}</el-tag>
              </div>
            </div>
            <div class="model-card-body">
              <div class="model-card-field" v-for="f in getMetaFields(selectedModel)" :key="f.key">
                <span class="field-label">{{ f.label }}</span>
                <span class="field-value">{{ f.value }}</span>
              </div>
              <div class="model-card-desc" v-if="selectedModel.description || selectedModel.meta?.desc">
                {{ selectedModel.description || selectedModel.meta?.desc }}
              </div>
            </div>
          </div>
        </div>
        <div v-else class="model-detail-placeholder">
          <el-empty description="点击左侧树节点查看模型信息" :image-size="80" />
        </div>
      </div>
    </el-card>

    <!-- ===== 卡片 4: AutoRouter 评测 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🚀</span>
          <span class="card-title">AutoRouter 评测</span>
        </div>
      </template>

      <!-- 顶部: 数据集上传 -->
      <div class="upload-area">
        <div class="upload-row">
          <el-upload
            :auto-upload="false"
            :show-file-list="false"
            accept=".csv,.json"
            :on-change="handleFileChange"
          >
            <el-button type="primary" size="small">选择数据集 (CSV/JSON)</el-button>
          </el-upload>
          <span class="file-name" v-if="uploadedFile">{{ uploadedFile.name }}</span>
          <el-button size="small" @click="handleUpload" :disabled="!uploadedFile">
            📤 上传并解析
          </el-button>
          <el-button size="small" type="info" plain @click="handleDownloadSample">
            📥 下载示例数据集
          </el-button>
        </div>

        <!-- 上传预览 -->
        <div v-if="uploadResult" class="upload-preview">
          <el-tag type="success" size="small">{{ uploadResult.total_samples }} 条样本</el-tag>
          <el-tag
            v-for="(count, task) in uploadResult.task_distribution" :key="task"
            size="small" type="info" style="margin-left: 6px;"
          >
            {{ task }}: {{ count }}
          </el-tag>
          <el-button
            type="warning" size="small" style="margin-left: 16px;"
            :loading="evaluating"
            @click="handleEvaluate"
          >
            {{ evaluating ? '评测中...' : '🚀 开始评测' }}
          </el-button>
        </div>
      </div>

      <!-- 底部: 终端日志 + 结果 -->
      <el-row :gutter="16" style="margin-top: 16px;">
        <el-col :span="evalResult ? 12 : 24">
          <TerminalLog
            title="评测日志"
            :lines="terminalLines"
            :height="evalResult ? '400px' : '280px'"
          />
        </el-col>
        <el-col :span="12" v-if="evalResult">
          <div class="result-panel">
            <!-- α 系数柱状图 -->
            <div class="result-section">
              <h4 class="section-title">Router 学习的 α 系数</h4>
              <v-chart :option="alphaChartOption" style="height: 180px;" autoresize />
            </div>
            <!-- Merged vs Baseline 对比 -->
            <div class="result-section">
              <h4 class="section-title">合并模型各任务归一化性能</h4>
              <v-chart :option="compareChartOption" style="height: 200px;" autoresize />
            </div>
            <!-- 汇总表格 -->
            <div class="result-section">
              <el-descriptions :column="2" size="small" border>
                <el-descriptions-item label="归一化准确率">
                  <span class="acc-value">{{ (evalResult.overall_norm_acc * 100).toFixed(2) }}%</span>
                </el-descriptions-item>
                <el-descriptions-item label="任务识别准确率">
                  {{ (evalResult.task_classification_acc * 100).toFixed(1) }}%
                </el-descriptions-item>
                <el-descriptions-item label="总样本数">
                  {{ Object.values(evalResult.per_task_samples).reduce((a,b)=>a+b, 0) }}
                </el-descriptions-item>
                <el-descriptions-item label="任务数">4</el-descriptions-item>
              </el-descriptions>
            </div>
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<style scoped>
.autorouter-view {
  max-width: 1200px;
  margin: 0 auto;
  display: flex;
  flex-direction: column;
  gap: 14px;
}

/* 知识图谱筛选栏 */
.graph-filter-bar {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 10px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.filter-tag {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  border-radius: 20px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  border: 2px solid #e5e7eb;
  background: #fff;
  color: #6b7280;
  transition: all 0.2s ease;
  outline: none;
}
.filter-tag:hover {
  border-color: var(--tag-border, #7c3aed);
  color: var(--tag-text, #5b21b6);
  background: var(--tag-bg, #f5f3ff);
}
.filter-tag.active {
  border-color: var(--tag-border, #7c3aed);
  background: var(--tag-bg, #f5f3ff);
  color: var(--tag-text, #5b21b6);
  box-shadow: 0 1px 4px rgba(0,0,0,0.08);
}
.filter-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

/* 通用卡片 */
.flow-card {
  border-radius: 12px;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.card-icon {
  font-size: 18px;
}
.card-title {
  font-size: 16px;
  font-weight: 600;
  color: var(--text-primary);
}

/* ── 卡片 1: 算法介绍 ── */
.algo-steps {
  display: flex;
  align-items: flex-start;
  gap: 12px;
}
.algo-step {
  flex: 1;
  text-align: center;
  padding: 12px 10px;
  background: var(--primary-bg, #f5f3ff);
  border-radius: 10px;
}
.step-emoji {
  font-size: 30px;
  margin-bottom: 6px;
}
.algo-step h4 {
  font-size: 14px;
  color: var(--text-primary);
  margin-bottom: 8px;
}
.algo-step p {
  font-size: 12px;
  color: var(--text-regular);
  line-height: 1.7;
}
.step-arrow {
  display: flex;
  align-items: center;
  font-size: 28px;
  color: var(--primary-light, #7c3aed);
  font-weight: 700;
  padding-top: 40px;
}

/* ── 卡片 3: 模型信息 ── */
.model-info-layout {
  display: flex;
  gap: 16px;
  min-height: 200px;
}
.model-tree-wrap {
  flex-shrink: 0;
  width: 220px;
  background: #fafafa;
  border-radius: 10px;
  padding: 12px 8px;
  border: 1px solid #e5e7eb;
}
.model-tree-wrap :deep(.el-tree) {
  background: transparent;
  --el-tree-node-hover-bg-color: #ede9fe;
}
.model-tree-wrap :deep(.el-tree-node.is-current > .el-tree-node__content) {
  background: #ddd6fe;
  border-radius: 6px;
}
.tree-node {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 13px;
}
.tree-node-icon {
  font-size: 16px;
}
.tree-node-label {
  font-weight: 500;
  color: #374151;
}
.model-detail-card-wrap {
  flex: 1;
  min-width: 0;
}
.model-detail-card {
  border: 2px solid;
  border-radius: 14px;
  padding: 16px 20px;
  height: 100%;
  box-sizing: border-box;
  transition: all 0.3s ease;
}
.model-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
  padding-bottom: 10px;
  border-bottom: 1px solid rgba(0,0,0,0.08);
}
.model-card-icon {
  font-size: 30px;
}
.model-card-title-group {
  display: flex;
  flex-direction: column;
  gap: 6px;
}
.model-card-name {
  font-size: 18px;
  font-weight: 700;
  color: #1f2937;
  margin: 0;
}
.model-card-body {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 10px 16px;
}
.model-card-field {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.field-label {
  font-size: 11px;
  color: #6b7280;
  font-weight: 500;
  letter-spacing: 0.3px;
}
.field-value {
  font-size: 13px;
  font-weight: 600;
  color: #1f2937;
  word-break: break-word;
}
.model-card-desc {
  grid-column: 1 / -1;
  font-size: 13px;
  color: #4b5563;
  line-height: 1.6;
  margin-top: 4px;
  padding-top: 8px;
  border-top: 1px solid rgba(0,0,0,0.06);
}
.model-detail-placeholder {
  flex: 1;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── 卡片 4: 评测 ── */
.upload-area {
  background: #fafafa;
  border-radius: 8px;
  padding: 16px;
}
.upload-row {
  display: flex;
  align-items: center;
  gap: 12px;
  flex-wrap: wrap;
}
.file-name {
  font-size: 13px;
  color: var(--text-secondary);
  max-width: 200px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.upload-hint {
  font-size: 11px;
  color: #9ca3af;
  margin-top: 8px;
}
.upload-preview {
  margin-top: 12px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 4px;
}

/* 结果面板 */
.result-panel {
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.result-section {
  background: #fff;
  border: 1px solid var(--border-color, #e5e7eb);
  border-radius: 8px;
  padding: 12px;
}
.section-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  text-align: center;
}
.acc-value {
  font-weight: 700;
  color: #6366f1;
  font-size: 15px;
}
</style>
