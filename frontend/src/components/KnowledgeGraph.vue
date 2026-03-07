<script setup>
import { ref, watch, onMounted, onUnmounted, shallowRef } from 'vue'
import * as echarts from 'echarts'

const props = defineProps({
  nodes: { type: Array, default: () => [] },
  edges: { type: Array, default: () => [] },
  height: { type: String, default: '400px' },
})

const chartRef = ref(null)
const chartInstance = shallowRef(null)

const NODE_COLORS = {
  pretrained: '#6366f1',
  module: '#10b981',
  finetuned: '#f59e0b',
  merged: '#ef4444',
  router: '#8b5cf6',
  task_vector: '#06b6d4',
  default: '#94a3b8',
}

const EDGE_STYLES = {
  dashed: [5, 5],
  dotted: [2, 4],
  solid: null,
}

const buildOption = () => {
  const echartNodes = props.nodes.map((n) => ({
    id: n.id,
    name: n.name,
    symbolSize: n.type === 'pretrained' ? 55 : n.type === 'merged' ? 50 : 42,
    category: n.type || 'default',
    itemStyle: {
      color: NODE_COLORS[n.type] || NODE_COLORS.default,
      borderColor: '#fff',
      borderWidth: 2,
      shadowBlur: 8,
      shadowColor: 'rgba(0,0,0,0.15)',
    },
    label: { show: true, fontSize: 11, color: '#1e1b4b' },
    tooltip: n.meta || '',
  }))

  const echartEdges = props.edges.map((e) => ({
    source: e.source,
    target: e.target,
    label: {
      show: true,
      formatter: e.relation || '',
      fontSize: 10,
      color: '#6b7280',
    },
    lineStyle: {
      color: '#a78bfa',
      width: e.style === 'solid' ? 2.5 : 1.5,
      type: EDGE_STYLES[e.style] ?? null,
      curveness: 0.2,
    },
  }))

  return {
    tooltip: {
      formatter: (params) => {
        if (params.dataType === 'node') {
          const meta = params.data.tooltip
          if (typeof meta === 'object' && meta) {
            return Object.entries(meta)
              .map(([k, v]) => `<b>${k}</b>: ${v}`)
              .join('<br/>')
          }
          return params.name
        }
        return params.data.label?.formatter || ''
      },
    },
    series: [
      {
        type: 'graph',
        layout: 'force',
        roam: true,
        draggable: true,
        force: {
          repulsion: 260,
          gravity: 0.1,
          edgeLength: [100, 200],
        },
        data: echartNodes,
        links: echartEdges,
        edgeSymbol: ['none', 'arrow'],
        edgeSymbolSize: 8,
        emphasis: {
          focus: 'adjacency',
          lineStyle: { width: 4 },
        },
      },
    ],
  }
}

const renderChart = () => {
  if (!chartRef.value) return
  if (!chartInstance.value) {
    chartInstance.value = echarts.init(chartRef.value)
  }
  chartInstance.value.setOption(buildOption(), true)
}

watch(() => [props.nodes, props.edges], renderChart, { deep: true })

onMounted(() => {
  renderChart()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  chartInstance.value?.dispose()
})

const handleResize = () => {
  chartInstance.value?.resize()
}
</script>

<template>
  <div ref="chartRef" class="knowledge-graph" :style="{ height }"></div>
</template>

<style scoped>
.knowledge-graph {
  width: 100%;
}
</style>
