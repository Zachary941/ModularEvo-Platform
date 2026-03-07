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

/* ======== 卡片 3: 模型信息 ======== */
const allModels = ref([])
const selectedModelId = ref('')
const selectedModel = computed(() => allModels.value.find(m => m.id === selectedModelId.value))

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
    allModels.value = modelRes.data.map(m => ({
      ...m,
      meta: {
        params: m.params,
        desc: m.description,
        ...(m.baseline_acc ? { acc: `${(m.baseline_acc*100).toFixed(1)}%` } : {}),
        ...(m.num_classes ? { classes: m.num_classes } : {}),
      },
    }))
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
      <KnowledgeGraph
        v-if="graphNodes.length"
        :nodes="graphNodes"
        :edges="graphEdges"
        height="420px"
      />
      <el-empty v-else description="加载中..." />
    </el-card>

    <!-- ===== 卡片 3: 模型信息 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">📦</span>
          <span class="card-title">模型信息</span>
        </div>
      </template>
      <div class="model-info-layout">
        <div class="model-list">
          <el-select v-model="selectedModelId" placeholder="选择模型" style="width: 220px;">
            <el-option
              v-for="m in allModels" :key="m.id"
              :value="m.id"
              :label="m.name"
            >
              <span class="model-type-dot" :class="m.type"></span>
              <span>{{ m.name }}</span>
            </el-option>
          </el-select>
        </div>
        <div class="model-detail" v-if="selectedModel">
          <el-descriptions :column="2" border size="small">
            <el-descriptions-item label="模型名称">{{ selectedModel.name }}</el-descriptions-item>
            <el-descriptions-item label="类型">
              <el-tag
                size="small"
                :type="selectedModel.type === 'pretrained' ? '' : selectedModel.type === 'finetuned' ? 'warning' : selectedModel.type === 'router' ? 'success' : selectedModel.type === 'merged' ? 'danger' : 'info'"
              >
                {{ {pretrained:'预训练', finetuned:'微调模型', task_vector:'任务向量', router:'路由网络', merged:'合并模型'}[selectedModel.type] || selectedModel.type }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="参数量" v-if="selectedModel.params">{{ selectedModel.params }}</el-descriptions-item>
            <el-descriptions-item label="基线ACC" v-if="selectedModel.baseline_acc">{{ (selectedModel.baseline_acc * 100).toFixed(2) }}%</el-descriptions-item>
            <el-descriptions-item label="类别数" v-if="selectedModel.num_classes">{{ selectedModel.num_classes }}</el-descriptions-item>
            <el-descriptions-item label="描述" v-if="selectedModel.description" :span="2">{{ selectedModel.description }}</el-descriptions-item>
          </el-descriptions>
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
  gap: 20px;
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
  padding: 16px 12px;
  background: var(--primary-bg, #f5f3ff);
  border-radius: 10px;
}
.step-emoji {
  font-size: 36px;
  margin-bottom: 8px;
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
  gap: 20px;
}
.model-list {
  flex-shrink: 0;
}
.model-type-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  margin-right: 4px;
  vertical-align: middle;
}
.model-type-dot.pretrained { background: #6366f1; }
.model-type-dot.finetuned { background: #f59e0b; }
.model-type-dot.task_vector { background: #06b6d4; }
.model-type-dot.router { background: #8b5cf6; }
.model-type-dot.merged { background: #ef4444; }
.model-detail {
  flex: 1;
  min-width: 0;
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
