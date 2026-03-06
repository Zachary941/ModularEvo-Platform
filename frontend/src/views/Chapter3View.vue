<script setup>
import { ref, onMounted, computed } from 'vue'
import { ElMessage } from 'element-plus'
import { use } from 'echarts/core'
import { CanvasRenderer } from 'echarts/renderers'
import { GraphChart, BarChart } from 'echarts/charts'
import {
  TitleComponent, TooltipComponent, LegendComponent,
  GridComponent
} from 'echarts/components'
import VChart from 'vue-echarts'
import {
  getModules, loadModule, getFinetuned,
  evaluateFinetuned, mergeModels, getGraph
} from '../api/chapter3'

use([
  CanvasRenderer, GraphChart, BarChart,
  TitleComponent, TooltipComponent, LegendComponent, GridComponent
])

// ──── 知识图谱 ────
const graphData = ref({ nodes: [], edges: [] })
const graphOption = computed(() => {
  if (!graphData.value.nodes.length) return {}
  const typeColor = {
    pretrained: '#409EFF', module: '#67C23A',
    finetuned: '#E6A23C', merged: '#F56C6C',
  }
  const styleMap = { dashed: [5, 5], dotted: [2, 3], solid: [0, 0] }
  return {
    tooltip: {
      formatter(p) {
        if (p.dataType === 'node') {
          const m = p.data.meta || {}
          const lines = [p.data.name]
          if (m.params) lines.push(`参数量: ${m.params}`)
          if (m.wrr) lines.push(`WRR: ${m.wrr}`)
          if (m.sparsity) lines.push(`稀疏率: ${m.sparsity}`)
          if (m.task) lines.push(`任务: ${m.task}`)
          if (m.desc) lines.push(m.desc)
          return lines.join('<br/>')
        }
        return p.data.relation || ''
      }
    },
    series: [{
      type: 'graph', layout: 'force', roam: true, draggable: true,
      symbolSize: 50, symbol: 'circle',
      label: { show: true, fontSize: 11 },
      edgeLabel: { show: true, formatter: '{c}', fontSize: 10 },
      force: { repulsion: 300, edgeLength: 150, gravity: 0.1 },
      lineStyle: { width: 2, curveness: 0.1 },
      emphasis: { focus: 'adjacency', lineStyle: { width: 4 } },
      data: graphData.value.nodes.map(n => ({
        name: n.name, id: n.id, meta: n.meta,
        itemStyle: { color: typeColor[n.type] || '#909399' },
        category: n.type,
      })),
      edges: graphData.value.edges.map(e => ({
        source: graphData.value.nodes.findIndex(n => n.id === e.source),
        target: graphData.value.nodes.findIndex(n => n.id === e.target),
        value: e.relation,
        lineStyle: {
          type: styleMap[e.style] || 'solid',
          color: e.relation === '模块化' ? '#67C23A'
            : e.relation === '模块微调' ? '#E6A23C'
            : e.relation === '合并' ? '#F56C6C' : '#409EFF',
        }
      })),
    }]
  }
})

// ──── Step 1: 模块化 ────
const modules = ref([])
const moduleLoading = ref({})
const loadedModules = ref({})  // language -> {sparsity, wrr, layer_stats}

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

function getLayerChartOption(lang) {
  const mod = loadedModules.value[lang]
  if (!mod) return {}
  const stats = mod.layer_stats
  // 只展示每层的保留率 (ratio)
  return {
    title: { text: `${lang} 模块 — 逐层权重保留率`, left: 'center', textStyle: { fontSize: 13 } },
    tooltip: {
      trigger: 'axis',
      formatter(params) {
        const p = params[0]
        return `${p.name}<br/>保留率: ${(p.value * 100).toFixed(2)}%`
      }
    },
    grid: { left: 50, right: 20, top: 40, bottom: 60 },
    xAxis: {
      type: 'category',
      data: stats.map((_, i) => `L${i}`),
      axisLabel: { rotate: 45, fontSize: 9 },
    },
    yAxis: { type: 'value', name: '保留率', max: 1, axisLabel: { formatter: v => `${(v * 100).toFixed(0)}%` } },
    series: [{
      type: 'bar', data: stats.map(s => s.ratio),
      itemStyle: { color: lang === 'java' ? '#409EFF' : '#67C23A' },
    }],
  }
}

// ──── Step 2: 微调模型 ────
const finetunedList = ref([])
const evalLoading = ref({})
const evalResults = ref({})  // task -> metrics

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

// ──── Step 3: 模型合并 ────
const mergeMethod = ref('task_arithmetic')
const scalingCoeffs = ref([0.5, 0.5])
const tiesMaskRate = ref(0.8)
const dareMaskRates = ref([0.5, 0.5])
const merging = ref(false)
const mergeResult = ref(null)

const methodOptions = [
  { value: 'task_arithmetic', label: 'Task Arithmetic' },
  { value: 'ties', label: 'TIES Merging' },
  { value: 'dare', label: 'DARE (Mask Merging)' },
]

async function handleMerge() {
  merging.value = true
  mergeResult.value = null
  try {
    const params = {
      method: mergeMethod.value,
      scaling_coefficients: scalingCoeffs.value,
    }
    if (mergeMethod.value === 'ties') {
      params.param_value_mask_rate = tiesMaskRate.value
    } else if (mergeMethod.value === 'dare') {
      params.weight_mask_rates = dareMaskRates.value
    }
    const { data } = await mergeModels(params)
    mergeResult.value = data
    ElMessage.success('模型合并完成')
  } catch (e) {
    ElMessage.error(`合并失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    merging.value = false
  }
}

const mergeChartOption = computed(() => {
  if (!mergeResult.value) return {}
  const r = mergeResult.value.results
  const tasks = Object.keys(r)
  const mainMetric = {
    clone_detection: 'eval_f1', code_search: 'f1',
  }
  return {
    title: { text: `合并结果 — ${mergeResult.value.method}`, left: 'center', textStyle: { fontSize: 13 } },
    tooltip: { trigger: 'axis' },
    grid: { left: 60, right: 20, top: 40, bottom: 40 },
    xAxis: {
      type: 'category',
      data: tasks.map(t => t === 'clone_detection' ? '克隆检测' : '代码搜索'),
    },
    yAxis: { type: 'value', name: 'F1', max: 1, axisLabel: { formatter: v => `${(v * 100).toFixed(0)}%` } },
    series: [{
      type: 'bar',
      data: tasks.map(t => r[t][mainMetric[t]] || 0),
      itemStyle: { color: '#E6A23C' },
      label: { show: true, position: 'top', formatter: p => `${(p.value * 100).toFixed(2)}%` },
    }],
  }
})

// ──── 初始化 ────
onMounted(async () => {
  try {
    const [modRes, ftRes, graphRes] = await Promise.all([
      getModules(), getFinetuned(), getGraph(),
    ])
    modules.value = modRes.data
    finetunedList.value = ftRes.data
    graphData.value = graphRes.data
  } catch (e) {
    console.error('初始化失败:', e)
  }
})
</script>

<template>
  <div class="chapter3-container">
    <el-page-header @back="$router.push('/')" title="返回首页">
      <template #content>
        <span class="page-title">第三章：CodeBERT 模块化全流程</span>
      </template>
    </el-page-header>

    <!-- 知识图谱 -->
    <el-card class="section-card" shadow="never">
      <template #header><span>知识图谱</span></template>
      <v-chart v-if="graphData.nodes.length" :option="graphOption" style="height: 320px;" autoresize />
      <el-empty v-else description="加载中..." />
    </el-card>

    <!-- 三步流程 -->
    <el-row :gutter="16">
      <!-- Step 1: 模块化 -->
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag>Step 1</el-tag>
              <span>模块化 (Modularization)</span>
            </div>
          </template>
          <div class="step-body">
            <p class="desc">加载预训练语言模块的稀疏 mask，查看各层权重保留率。</p>
            <div v-for="mod in modules" :key="mod.language" class="module-item">
              <div class="module-row">
                <el-tag :type="mod.exists ? 'success' : 'danger'" size="small">
                  {{ mod.language }}
                </el-tag>
                <span class="wrr-label">WRR: {{ mod.wrr_label }}</span>
                <el-button
                  size="small" type="primary"
                  :loading="moduleLoading[mod.language]"
                  :disabled="!!loadedModules[mod.language]"
                  @click="handleLoadModule(mod.language)"
                >
                  {{ loadedModules[mod.language] ? '已加载' : '加载模块' }}
                </el-button>
              </div>
              <!-- 逐层统计图 -->
              <div v-if="loadedModules[mod.language]" class="chart-container">
                <div class="stat-summary">
                  <el-tag type="info" size="small">稀疏率: {{ loadedModules[mod.language].sparsity }}%</el-tag>
                  <el-tag type="success" size="small">WRR: {{ loadedModules[mod.language].wrr }}%</el-tag>
                  <el-tag size="small">层数: {{ loadedModules[mod.language].layer_stats.length }}</el-tag>
                </div>
                <v-chart :option="getLayerChartOption(mod.language)" style="height: 200px;" autoresize />
              </div>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- Step 2: 下游任务微调 -->
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag type="success">Step 2</el-tag>
              <span>下游任务微调</span>
            </div>
          </template>
          <div class="step-body">
            <p class="desc">加载预微调模型并评测下游任务性能。</p>
            <div v-for="ft in finetunedList" :key="ft.task" class="ft-item">
              <div class="ft-row">
                <div>
                  <strong>{{ ft.task === 'clone_detection' ? '克隆检测' : '代码搜索' }}</strong>
                  <el-tag size="small" type="info" style="margin-left: 6px;">{{ ft.params_m }}M</el-tag>
                </div>
                <el-button
                  size="small" type="success"
                  :loading="evalLoading[ft.task]"
                  :disabled="!!evalResults[ft.task]"
                  @click="handleEvaluate(ft.task)"
                >
                  {{ evalResults[ft.task] ? '已评测' : '执行评测' }}
                </el-button>
              </div>
              <!-- 评测结果 -->
              <el-descriptions
                v-if="evalResults[ft.task]"
                :column="2" size="small" border
                class="eval-table"
              >
                <el-descriptions-item
                  v-for="(val, key) in evalResults[ft.task]" :key="key"
                  :label="key"
                >
                  {{ (val * 100).toFixed(2) }}%
                </el-descriptions-item>
              </el-descriptions>
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- Step 3: 模型合并 -->
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag type="warning">Step 3</el-tag>
              <span>模型合并</span>
            </div>
          </template>
          <div class="step-body">
            <p class="desc">选择合并方法和参数，合并两个微调模型并评测。</p>
            <el-form label-position="top" size="small">
              <el-form-item label="合并方法">
                <el-select v-model="mergeMethod" style="width: 100%;">
                  <el-option
                    v-for="opt in methodOptions" :key="opt.value"
                    :value="opt.value" :label="opt.label"
                  />
                </el-select>
              </el-form-item>
              <el-form-item label="缩放系数 (α₁, α₂)">
                <el-row :gutter="8">
                  <el-col :span="12">
                    <el-input-number v-model="scalingCoeffs[0]" :step="0.1" :min="0" :max="2" :precision="2" size="small" style="width: 100%;" />
                  </el-col>
                  <el-col :span="12">
                    <el-input-number v-model="scalingCoeffs[1]" :step="0.1" :min="0" :max="2" :precision="2" size="small" style="width: 100%;" />
                  </el-col>
                </el-row>
              </el-form-item>
              <el-form-item v-if="mergeMethod === 'ties'" label="TIES mask rate">
                <el-slider v-model="tiesMaskRate" :min="0" :max="1" :step="0.05" show-input />
              </el-form-item>
              <el-form-item v-if="mergeMethod === 'dare'" label="DARE mask rates">
                <el-row :gutter="8">
                  <el-col :span="12">
                    <el-input-number v-model="dareMaskRates[0]" :step="0.1" :min="0" :max="1" :precision="2" size="small" style="width: 100%;" />
                  </el-col>
                  <el-col :span="12">
                    <el-input-number v-model="dareMaskRates[1]" :step="0.1" :min="0" :max="1" :precision="2" size="small" style="width: 100%;" />
                  </el-col>
                </el-row>
              </el-form-item>
              <el-form-item>
                <el-button type="warning" :loading="merging" @click="handleMerge" style="width: 100%;">
                  {{ merging ? '合并中...' : '开始合并 & 评测' }}
                </el-button>
              </el-form-item>
            </el-form>
            <!-- 合并结果 -->
            <div v-if="mergeResult">
              <v-chart :option="mergeChartOption" style="height: 220px;" autoresize />
              <el-descriptions :column="1" size="small" border class="merge-detail">
                <template v-for="(metrics, task) in mergeResult.results" :key="task">
                  <el-descriptions-item :label="task === 'clone_detection' ? '克隆检测' : '代码搜索'">
                    <span v-for="(val, key) in metrics" :key="key" class="metric-tag">
                      {{ key }}: {{ (val * 100).toFixed(2) }}%
                    </span>
                  </el-descriptions-item>
                </template>
              </el-descriptions>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.chapter3-container {
  max-width: 1400px;
  margin: 0 auto;
  padding: 20px;
}
.page-title {
  font-size: 18px;
  font-weight: 600;
}
.section-card {
  margin: 20px 0;
}
.step-card {
  min-height: 400px;
}
.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
.step-body {
  font-size: 13px;
}
.desc {
  color: #666;
  margin-bottom: 12px;
  font-size: 12px;
}

/* Step 1 */
.module-item {
  margin-bottom: 16px;
}
.module-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}
.wrr-label {
  color: #999;
  font-size: 12px;
  flex: 1;
}
.stat-summary {
  display: flex;
  gap: 6px;
  margin-bottom: 4px;
}
.chart-container {
  margin-top: 8px;
}

/* Step 2 */
.ft-item {
  margin-bottom: 16px;
}
.ft-row {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}
.eval-table {
  margin-top: 8px;
}

/* Step 3 */
.merge-detail {
  margin-top: 8px;
}
.metric-tag {
  margin-right: 8px;
  font-size: 12px;
  color: #606266;
}
</style>
