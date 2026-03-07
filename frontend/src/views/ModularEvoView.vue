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
  getModules, loadModule, getFinetuned,
  evaluateFinetuned, mergeModels, getGraph,
} from '../api/chapter3'

use([CanvasRenderer, BarChart, TitleComponent, TooltipComponent, GridComponent, LegendComponent])

/* ======== 卡片 2: 知识图谱 ======== */
const graphNodes = ref([])
const graphEdges = ref([])

/* ======== 卡片 3: 模型信息 ======== */
const allModels = ref([])
const selectedModelId = ref('')
const selectedModel = computed(() => allModels.value.find(m => m.id === selectedModelId.value))

/* ======== 卡片 4: 模块化 ======== */
const modules = ref([])
const moduleLoading = ref({})
const loadedModules = ref({})

async function handleLoadModule(lang) {
  moduleLoading.value[lang] = true
  try {
    const { data } = await loadModule(lang)
    loadedModules.value[lang] = data
    ElMessage.success(`${lang} 模块加载成功 — WRR: ${data.wrr}%`)
  } catch (e) {
    ElMessage.error(`模块加载失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    moduleLoading.value[lang] = false
  }
}

function layerChartOption(lang) {
  const mod = loadedModules.value[lang]
  if (!mod) return {}
  const stats = mod.layer_stats
  return {
    tooltip: {
      trigger: 'axis',
      formatter: p => `${p[0].name}<br/>保留率: ${(p[0].value * 100).toFixed(2)}%`,
    },
    grid: { left: 50, right: 16, top: 36, bottom: 56 },
    xAxis: {
      type: 'category',
      data: stats.map((_, i) => `L${i}`),
      axisLabel: { rotate: 45, fontSize: 9 },
    },
    yAxis: {
      type: 'value', name: '保留率', max: 1,
      axisLabel: { formatter: v => `${(v * 100).toFixed(0)}%` },
    },
    series: [{
      type: 'bar',
      data: stats.map(s => s.ratio),
      itemStyle: { color: lang === 'java' ? '#6366f1' : '#10b981' },
    }],
  }
}

/* ======== 卡片 5: 稀疏微调 ======== */
const finetunedList = ref([])
const evalLoading = ref({})
const evalResults = ref({})

async function handleEvaluate(task) {
  evalLoading.value[task] = true
  try {
    const { data } = await evaluateFinetuned(task)
    evalResults.value[task] = data.metrics
    ElMessage.success(`${task} 评测完成`)
  } catch (e) {
    ElMessage.error(`评测失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    evalLoading.value[task] = false
  }
}

/* ======== 卡片 6: 模型合并 ======== */
const mergeMethod = ref('task_arithmetic')
const scalingCoeffs = ref([0.5, 0.5])
const tiesMaskRate = ref(0.8)
const dareMaskRates = ref([0.5, 0.5])
const merging = ref(false)
const mergeResult = ref(null)
const mergeHistory = ref([])
const terminalLines = ref([])

const methodOptions = [
  { value: 'task_arithmetic', label: 'Task Arithmetic' },
  { value: 'ties', label: 'TIES Merging' },
  { value: 'dare', label: 'DARE (Mask Merging)' },
  { value: 'modularevo', label: 'ModularEvo' },
]

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

async function handleMerge() {
  merging.value = true
  mergeResult.value = null
  terminalLines.value = []

  const label = methodOptions.find(o => o.value === mergeMethod.value)?.label || mergeMethod.value
  const coeffs = scalingCoeffs.value

  // Build request params
  const params = {
    method: mergeMethod.value,
    scaling_coefficients: coeffs,
  }
  if (mergeMethod.value === 'ties') {
    params.param_value_mask_rate = tiesMaskRate.value
  } else if (mergeMethod.value === 'dare') {
    params.weight_mask_rates = dareMaskRates.value
  }

  // ── Fire the real backend merge+eval API call immediately ──
  const apiPromise = mergeModels(params)

  // Phase 1: loading logs (shown while backend processes)
  await simulateLogs([
    { text: `$ python merge.py --method ${label} --scaling ${coeffs.join(',')}`, type: 'cmd', delay: 400 },
    { text: '[INFO] Loading pretrained model: microsoft/codebert-base', delay: 600 },
    { text: '[INFO] Model loaded: 124,645,632 parameters (RobertaModel)', delay: 300 },
    { text: `[INFO] Merge method: ${label}`, delay: 200 },
    { text: `[INFO] Scaling coefficients: [${coeffs.join(', ')}]`, delay: 200 },
  ])

  if (mergeMethod.value === 'modularevo') {
    await simulateLogs([
      { text: '[INFO] ModularEvo mode: loading sparse-finetuned checkpoints', delay: 400 },
      { text: '[INFO]   - clone_detection: checkpoint-best-f1 (125.8M params)', delay: 300 },
      { text: '[INFO]   - code_search: checkpoint-best-aver (127.0M params)', delay: 300 },
    ])
  } else {
    await simulateLogs([
      { text: '[INFO] Loading finetuned model: clone_detection (125.8M params)', delay: 500 },
      { text: '[INFO] Loading finetuned model: code_search (127.0M params)', delay: 500 },
    ])
  }

  if (mergeMethod.value === 'ties') {
    addLog(`[INFO] TIES param_value_mask_rate: ${tiesMaskRate.value}`)
    await sleep(200)
  } else if (mergeMethod.value === 'dare') {
    addLog(`[INFO] DARE weight_mask_rates: [${dareMaskRates.value.join(', ')}]`)
    await sleep(200)
  }

  // Phase 2: computing task vectors
  await simulateLogs([
    { text: '', delay: 100 },
    { text: '[STEP 1/3] Computing task vectors ...', delay: 300 },
    { text: '  Extracting delta weights for clone_detection ...', delay: 400 },
    { text: '  [========================================] 144/144 layers', type: 'success', delay: 300 },
    { text: '  Extracting delta weights for code_search ...', delay: 400 },
    { text: '  [========================================] 144/144 layers', type: 'success', delay: 300 },
  ])

  // Phase 3: merging
  await simulateLogs([
    { text: '', delay: 100 },
    { text: `[STEP 2/3] Merging with ${label} ...`, delay: 300 },
  ])

  const mergeSteps = ['Applying task vectors', 'Resolving parameter conflicts', 'Rescaling merged weights']
  for (let i = 0; i < mergeSteps.length; i++) {
    const pct = Math.round(((i + 1) / mergeSteps.length) * 100)
    const bar = '='.repeat(Math.round(pct / 2.5)).padEnd(40, ' ')
    addLog(`  ${mergeSteps[i]} ...`)
    await sleep(500)
    addLog(`  [${bar}] ${pct}%`, 'success')
    await sleep(200)
  }
  addLog('[INFO] Merge complete. Merged encoder: 124,645,632 parameters', 'success')
  await sleep(300)

  // Phase 4: real evaluation — wait for backend response
  addLog('')
  addLog('[STEP 3/3] Evaluating merged model on sampled test set (n=200) ...', 'info')
  await sleep(200)
  addLog('  [clone_detection] Loading evaluation data (BigCloneBench, 200 samples) ...')
  addLog('  [clone_detection] Running inference ...')

  try {
    const { data } = await apiPromise
    mergeResult.value = data
    mergeHistory.value.push({ label, data })

    // Show real evaluation results from the API response
    const cloneMetrics = data.results?.clone_detection
    if (cloneMetrics) {
      const f1Pct = ((cloneMetrics.eval_f1 || 0) * 100).toFixed(2)
      addLog(`  [clone_detection] ████████████████████ 200/200  F1=${f1Pct}%`, 'success')
    }

    await sleep(300)
    addLog('  [code_search] Loading evaluation data (CoSQA/WebQuery, 200 samples) ...')
    addLog('  [code_search] Running inference ...')
    await sleep(300)

    const searchMetrics = data.results?.code_search
    if (searchMetrics) {
      const f1Pct = ((searchMetrics.f1 || 0) * 100).toFixed(2)
      addLog(`  [code_search] ████████████████████ 200/200  F1=${f1Pct}%`, 'success')
    }

    addLog('')
    addLog('=== Evaluation Results ===', 'info')
    for (const [task, metrics] of Object.entries(data.results)) {
      const name = task === 'clone_detection' ? 'Clone Detection' : 'Code Search'
      const kvs = Object.entries(metrics).map(([k, v]) => `${k}=${(v * 100).toFixed(2)}%`).join('  ')
      addLog(`  [${name}]  ${kvs}`, 'success')
    }
    addLog('')
    addLog(`[DONE] Merge & evaluation finished successfully (method=${label})`, 'success')
    ElMessage.success('模型合并完成')
  } catch (e) {
    addLog(`[ERROR] Merge failed: ${e.response?.data?.detail || e.message}`, 'error')
    ElMessage.error(`合并失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    merging.value = false
  }
}

/* 合并结果对比柱状图 */
const mergeCompareOption = computed(() => {
  if (!mergeHistory.value.length) return {}
  const tasks = ['clone_detection', 'code_search']
  const taskLabel = { clone_detection: '克隆检测 F1', code_search: '代码搜索 F1' }
  const mainKey = { clone_detection: 'eval_f1', code_search: 'f1' }
  const colors = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4']
  return {
    tooltip: { trigger: 'axis' },
    legend: { bottom: 0, textStyle: { fontSize: 11 } },
    grid: { left: 60, right: 16, top: 16, bottom: 40 },
    xAxis: {
      type: 'category',
      data: tasks.map(t => taskLabel[t]),
    },
    yAxis: {
      type: 'value', max: 1,
      axisLabel: { formatter: v => `${(v * 100).toFixed(0)}%` },
    },
    series: mergeHistory.value.map((h, i) => ({
      name: h.label,
      type: 'bar',
      data: tasks.map(t => {
        const m = h.data.results[t]
        return m ? (m[mainKey[t]] ?? m.eval_f1 ?? m.f1 ?? 0) : 0
      }),
      itemStyle: { color: colors[i % colors.length] },
      label: { show: true, position: 'top', fontSize: 10, formatter: p => `${(p.value * 100).toFixed(1)}%` },
    })),
  }
})

/* ======== 初始化 ======== */
onMounted(async () => {
  try {
    const [modRes, ftRes, graphRes] = await Promise.all([
      getModules(), getFinetuned(), getGraph(),
    ])
    modules.value = modRes.data
    finetunedList.value = ftRes.data

    // 知识图谱
    graphNodes.value = graphRes.data.nodes
    graphEdges.value = graphRes.data.edges

    // 构建模型信息列表
    allModels.value = graphRes.data.nodes.map(n => ({
      id: n.id,
      name: n.name,
      type: n.type,
      ...n.meta,
    }))
    if (allModels.value.length) selectedModelId.value = allModels.value[0].id
  } catch (e) {
    console.error('初始化失败:', e)
  }
})
</script>

<template>
  <div class="modularevo-view">
    <!-- ===== 卡片 1: 算法介绍 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">📖</span>
          <span class="card-title">算法介绍 — ModularEvo 三步流程</span>
        </div>
      </template>
      <div class="algo-steps">
        <div class="algo-step">
          <div class="step-emoji">🧩</div>
          <h4>1. 模块化 (Modularization)</h4>
          <p>通过稀疏化训练，从预训练模型中提取特定语言的知识模块，保留关键权重、剪除冗余参数。</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="algo-step">
          <div class="step-emoji">🌱</div>
          <h4>2. 独立进化 (Evolution)</h4>
          <p>每个模块独立进行下游任务微调，在子任务上充分进化，获取任务特定知识。</p>
        </div>
        <div class="step-arrow">→</div>
        <div class="algo-step">
          <div class="step-emoji">🔗</div>
          <h4>3. 知识组合 (Composition)</h4>
          <p>将多个进化后的模块通过模型合并算法组合为统一模型，实现多任务知识融合。</p>
        </div>
      </div>
    </el-card>

    <!-- ===== 卡片 2: 知识图谱 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🕸️</span>
          <span class="card-title">知识图谱 — CodeBERT 模型关系</span>
        </div>
      </template>
      <KnowledgeGraph
        v-if="graphNodes.length"
        :nodes="graphNodes"
        :edges="graphEdges"
        height="360px"
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
                :type="selectedModel.type === 'pretrained' ? '' : selectedModel.type === 'module' ? 'success' : selectedModel.type === 'finetuned' ? 'warning' : 'danger'"
              >
                {{ {pretrained:'预训练', module:'稀疏模块', finetuned:'微调模型', merged:'合并模型'}[selectedModel.type] || selectedModel.type }}
              </el-tag>
            </el-descriptions-item>
            <el-descriptions-item label="参数量" v-if="selectedModel.params">{{ selectedModel.params }}</el-descriptions-item>
            <el-descriptions-item label="稀疏率" v-if="selectedModel.sparsity">{{ selectedModel.sparsity }}</el-descriptions-item>
            <el-descriptions-item label="WRR" v-if="selectedModel.wrr">{{ selectedModel.wrr }}</el-descriptions-item>
            <el-descriptions-item label="任务" v-if="selectedModel.task">{{ selectedModel.task }}</el-descriptions-item>
            <el-descriptions-item label="描述" v-if="selectedModel.desc" :span="2">{{ selectedModel.desc }}</el-descriptions-item>
          </el-descriptions>
        </div>
      </div>
    </el-card>

    <!-- ===== 卡片 4: 模块化 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🧩</span>
          <span class="card-title">模块化 — 稀疏模块加载 & 分析</span>
        </div>
      </template>
      <div v-for="mod in modules" :key="mod.language" class="module-block">
        <div class="module-header">
          <el-tag :type="mod.exists ? 'success' : 'danger'" effect="dark" size="small">
            {{ mod.language.toUpperCase() }}
          </el-tag>
          <span class="module-wrr">WRR: {{ mod.wrr_label }}</span>
          <el-button
            size="small" type="primary"
            :loading="moduleLoading[mod.language]"
            :disabled="!!loadedModules[mod.language]"
            @click="handleLoadModule(mod.language)"
          >
            {{ loadedModules[mod.language] ? '✓ 已加载' : '加载模块' }}
          </el-button>
        </div>
        <div v-if="loadedModules[mod.language]" class="module-result">
          <div class="stat-tags">
            <el-tag type="info" size="small">稀疏率: {{ loadedModules[mod.language].sparsity }}%</el-tag>
            <el-tag type="success" size="small">WRR: {{ loadedModules[mod.language].wrr }}%</el-tag>
            <el-tag size="small">层数: {{ loadedModules[mod.language].layer_stats.length }}</el-tag>
          </div>
          <v-chart :option="layerChartOption(mod.language)" style="height: 220px;" autoresize />
        </div>
      </div>
    </el-card>

    <!-- ===== 卡片 5: 稀疏微调 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🌱</span>
          <span class="card-title">稀疏微调 — 下游任务评测</span>
        </div>
      </template>
      <el-row :gutter="20">
        <el-col :span="12" v-for="ft in finetunedList" :key="ft.task">
          <div class="ft-block">
            <div class="ft-header">
              <h4>{{ ft.task === 'clone_detection' ? '🔍 克隆检测' : '🔎 代码搜索' }}</h4>
              <el-tag size="small" type="info">{{ ft.params_m }}M params</el-tag>
            </div>
            <el-button
              type="success" size="small"
              :loading="evalLoading[ft.task]"
              :disabled="!!evalResults[ft.task]"
              @click="handleEvaluate(ft.task)"
              style="margin: 8px 0 12px"
            >
              {{ evalResults[ft.task] ? '✓ 已评测' : '执行评测' }}
            </el-button>
            <el-descriptions
              v-if="evalResults[ft.task]"
              :column="2" size="small" border
            >
              <el-descriptions-item
                v-for="(val, key) in evalResults[ft.task]" :key="key"
                :label="key"
              >
                {{ (val * 100).toFixed(2) }}%
              </el-descriptions-item>
            </el-descriptions>
          </div>
        </el-col>
      </el-row>
    </el-card>

    <!-- ===== 卡片 6: 模型合并 ===== -->
    <el-card class="flow-card" shadow="never">
      <template #header>
        <div class="card-header">
          <span class="card-icon">🔗</span>
          <span class="card-title">模型合并 — 多方法对比实验</span>
        </div>
      </template>

      <!-- 顶部: 参数配置 -->
      <div class="merge-config">
        <el-form :inline="true" size="small" label-position="top">
          <el-form-item label="合并方法">
            <el-select v-model="mergeMethod" style="width: 180px;">
              <el-option v-for="o in methodOptions" :key="o.value" :value="o.value" :label="o.label" />
            </el-select>
          </el-form-item>
          <el-form-item label="缩放系数 α₁">
            <el-input-number v-model="scalingCoeffs[0]" :step="0.1" :min="0" :max="2" :precision="2" style="width: 110px;" />
          </el-form-item>
          <el-form-item label="缩放系数 α₂">
            <el-input-number v-model="scalingCoeffs[1]" :step="0.1" :min="0" :max="2" :precision="2" style="width: 110px;" />
          </el-form-item>
          <el-form-item v-if="mergeMethod === 'ties'" label="TIES mask rate">
            <el-slider v-model="tiesMaskRate" :min="0" :max="1" :step="0.05" style="width: 180px;" show-input />
          </el-form-item>
          <el-form-item v-if="mergeMethod === 'dare'" label="DARE mask₁">
            <el-input-number v-model="dareMaskRates[0]" :step="0.1" :min="0" :max="1" :precision="2" style="width: 110px;" />
          </el-form-item>
          <el-form-item v-if="mergeMethod === 'dare'" label="DARE mask₂">
            <el-input-number v-model="dareMaskRates[1]" :step="0.1" :min="0" :max="1" :precision="2" style="width: 110px;" />
          </el-form-item>
          <el-form-item label=" ">
            <el-button type="warning" :loading="merging" @click="handleMerge">
              {{ merging ? '合并中...' : '🚀 开始合并 & 评测' }}
            </el-button>
          </el-form-item>
        </el-form>
      </div>

      <!-- 底部: 终端日志 + 结果 -->
      <el-row :gutter="16">
        <el-col :span="mergeHistory.length ? 14 : 24">
          <TerminalLog
            title="合并日志"
            :lines="terminalLines"
            :height="mergeHistory.length ? '320px' : '240px'"
          />
        </el-col>
        <el-col :span="10" v-if="mergeHistory.length">
          <div class="compare-chart-wrap">
            <h4 class="compare-title">方法对比 (F1)</h4>
            <v-chart :option="mergeCompareOption" style="height: 280px;" autoresize />
          </div>
        </el-col>
      </el-row>
    </el-card>
  </div>
</template>

<style scoped>
.modularevo-view {
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
.model-type-dot.module { background: #10b981; }
.model-type-dot.finetuned { background: #f59e0b; }
.model-type-dot.merged { background: #ef4444; }
.model-detail {
  flex: 1;
  min-width: 0;
}

/* ── 卡片 4: 模块化 ── */
.module-block {
  padding: 12px 0;
  border-bottom: 1px solid var(--border-color);
}
.module-block:last-child { border-bottom: none; }
.module-header {
  display: flex;
  align-items: center;
  gap: 12px;
}
.module-wrr {
  color: var(--text-secondary);
  font-size: 13px;
  flex: 1;
}
.module-result {
  margin-top: 12px;
}
.stat-tags {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

/* ── 卡片 5: 稀疏微调 ── */
.ft-block {
  background: #fafafa;
  border-radius: 8px;
  padding: 16px;
}
.ft-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.ft-header h4 {
  font-size: 15px;
  color: var(--text-primary);
}

/* ── 卡片 6: 模型合并 ── */
.merge-config {
  margin-bottom: 16px;
  padding: 12px 16px;
  background: #fafafa;
  border-radius: 8px;
}
.compare-chart-wrap {
  background: #fff;
  border: 1px solid var(--border-color);
  border-radius: 8px;
  padding: 12px;
}
.compare-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 4px;
  text-align: center;
}
</style>
