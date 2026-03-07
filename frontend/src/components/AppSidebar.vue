<script setup>
import { computed, ref, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import api from '../api/index.js'

const route = useRoute()

const isModularEvoActive = computed(() => route.path === '/modularevo')
const isAutoRouterActive = computed(() => route.path === '/autorouter')

const status = ref(null)

let timer = null

const fetchStatus = async () => {
  try {
    const { data } = await api.get('/system/status')
    status.value = data
  } catch {
    status.value = null
  }
}

const gpuUtil = computed(() => status.value ? `${status.value.gpu_util}%` : '--')
const gpuTemp = computed(() => status.value ? `${status.value.gpu_temp}°C` : '--')
const gpuMem = computed(() => {
  if (!status.value) return '--'
  const used = (status.value.mem_used_mb / 1024).toFixed(1)
  const total = (status.value.mem_total_mb / 1024).toFixed(0)
  return `${used} / ${total} GB`
})
const gpuName = computed(() => status.value?.gpu_name || '--')
const sysMem = computed(() => {
  if (!status.value) return '--'
  const used = (status.value.sys_mem_used_mb / 1024).toFixed(1)
  const total = (status.value.sys_mem_total_mb / 1024).toFixed(0)
  return `${used} / ${total} GB`
})
const cpuCount = computed(() => status.value ? `${status.value.cpu_count} 核` : '--')
const disk = computed(() => {
  if (!status.value) return '--'
  return `${status.value.disk_used_gb} / ${status.value.disk_total_gb} GB`
})
const pyVer = computed(() => status.value?.python_version || '--')
const uptime = computed(() => status.value?.uptime || '--')

const gpuUtilPercent = computed(() => status.value?.gpu_util ?? 0)
const sysMemPercent = computed(() => {
  if (!status.value || !status.value.sys_mem_total_mb) return 0
  return Math.round(status.value.sys_mem_used_mb / status.value.sys_mem_total_mb * 100)
})

onMounted(() => {
  fetchStatus()
  timer = setInterval(fetchStatus, 5000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<template>
  <div class="app-sidebar">
    <!-- 运行任务 -->
    <div class="sidebar-section">
      <div class="section-title">🚀 运行任务</div>
      <div class="task-list">
        <div class="task-item" :class="{ active: isModularEvoActive }">
          <div class="task-row">
            <span class="task-icon" :class="{ spinning: isModularEvoActive }">⚙️</span>
            <span class="task-name">ModularEvo 进化</span>
          </div>
          <el-tag
            :type="isModularEvoActive ? 'warning' : 'info'"
            size="small"
            effect="plain"
            class="task-tag"
          >
            {{ isModularEvoActive ? '⏳ 使用中' : '💤 未使用' }}
          </el-tag>
        </div>
        <div class="task-item" :class="{ active: isAutoRouterActive }">
          <div class="task-row">
            <span class="task-icon" :class="{ spinning: isAutoRouterActive }">⚙️</span>
            <span class="task-name">AutoRouter 组合</span>
          </div>
          <el-tag
            :type="isAutoRouterActive ? 'warning' : 'info'"
            size="small"
            effect="plain"
            class="task-tag"
          >
            {{ isAutoRouterActive ? '⏳ 使用中' : '💤 未使用' }}
          </el-tag>
        </div>
      </div>
    </div>

    <!-- GPU 状态 -->
    <div class="sidebar-section">
      <div class="section-title">🎮 GPU 状态</div>
      <div class="gpu-name">{{ gpuName }}</div>
      <div class="progress-row">
        <span class="progress-label">利用率</span>
        <el-progress
          :percentage="gpuUtilPercent"
          :stroke-width="8"
          :color="gpuUtilPercent > 80 ? '#ef4444' : gpuUtilPercent > 50 ? '#f59e0b' : '#10b981'"
          class="mini-progress"
        />
      </div>
      <div class="status-list">
        <div class="status-item">
          <span class="status-label">🌡️ 温度</span>
          <span class="status-value">{{ gpuTemp }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">💾 显存</span>
          <span class="status-value">{{ gpuMem }}</span>
        </div>
      </div>
    </div>

    <!-- 系统资源 -->
    <div class="sidebar-section">
      <div class="section-title">📊 系统资源</div>
      <div class="progress-row">
        <span class="progress-label">内存</span>
        <el-progress
          :percentage="sysMemPercent"
          :stroke-width="8"
          :color="sysMemPercent > 80 ? '#ef4444' : sysMemPercent > 50 ? '#f59e0b' : '#6366f1'"
          class="mini-progress"
        />
      </div>
      <div class="status-list">
        <div class="status-item">
          <span class="status-label">🧠 内存</span>
          <span class="status-value">{{ sysMem }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">⚡ CPU</span>
          <span class="status-value">{{ cpuCount }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">💿 磁盘</span>
          <span class="status-value">{{ disk }}</span>
        </div>
      </div>
    </div>

    <!-- 环境信息 -->
    <div class="sidebar-section">
      <div class="section-title">🔧 环境</div>
      <div class="status-list">
        <div class="status-item">
          <span class="status-label">🐍 Python</span>
          <span class="status-value">{{ pyVer }}</span>
        </div>
        <div class="status-item">
          <span class="status-label">⏱️ 运行时间</span>
          <span class="status-value">{{ uptime }}</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.app-sidebar {
  position: fixed;
  top: var(--header-height);
  left: 0;
  width: var(--sidebar-width);
  height: calc(100vh - var(--header-height));
  background: linear-gradient(180deg, var(--bg-sidebar) 0%, #faf5ff 100%);
  border-right: 1px solid var(--border-color);
  padding: 14px 10px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  overflow-y: auto;
  z-index: 90;
}

.sidebar-section {
  background: var(--bg-card);
  border: 1px solid var(--border-color);
  border-radius: 12px;
  padding: 12px;
  transition: box-shadow 0.2s ease;
}
.sidebar-section:hover {
  box-shadow: 0 2px 8px rgba(91, 33, 182, 0.06);
}

.section-title {
  font-size: 12px;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  margin-bottom: 10px;
}

/* ---- 运行任务 ---- */
.task-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.task-item {
  padding: 8px 10px;
  border-radius: 8px;
  transition: background 0.2s;
  border: 1px solid transparent;
}

.task-item.active {
  background: #fef9c3;
  border-color: #fde68a;
}

.task-row {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 4px;
}

.task-icon {
  font-size: 15px;
  flex-shrink: 0;
  display: inline-block;
  filter: hue-rotate(200deg) saturate(1.5);
}

.task-item.active .task-icon {
  filter: none;
}

.task-icon.spinning {
  animation: spin 2s linear infinite;
}

@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}

.task-name {
  color: var(--text-primary);
  font-weight: 600;
  font-size: 13px;
}

.task-tag {
  margin-left: 22px;
}

/* ---- GPU / 系统状态 ---- */
.gpu-name {
  font-size: 11px;
  color: var(--text-secondary);
  margin-bottom: 8px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.progress-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 8px;
}

.progress-label {
  font-size: 11px;
  color: var(--text-secondary);
  flex-shrink: 0;
  width: 32px;
}

.mini-progress {
  flex: 1;
}

.mini-progress :deep(.el-progress__text) {
  font-size: 11px !important;
  min-width: 32px;
}

.status-list {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
}

.status-label {
  color: var(--text-regular);
}

.status-value {
  color: var(--text-primary);
  font-weight: 600;
  font-variant-numeric: tabular-nums;
  font-size: 11px;
}

/* 自定义滚动条 */
.app-sidebar::-webkit-scrollbar {
  width: 4px;
}
.app-sidebar::-webkit-scrollbar-thumb {
  background: rgba(124, 58, 237, 0.18);
  border-radius: 4px;
}
</style>
