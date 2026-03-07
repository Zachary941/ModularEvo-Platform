<script setup>
import { ref, watch, nextTick } from 'vue'

const props = defineProps({
  lines: { type: Array, default: () => [] },
  title: { type: String, default: 'Terminal' },
  height: { type: String, default: '300px' },
})

const terminalRef = ref(null)

watch(
  () => props.lines.length,
  () => {
    nextTick(() => {
      if (terminalRef.value) {
        terminalRef.value.scrollTop = terminalRef.value.scrollHeight
      }
    })
  }
)
</script>

<template>
  <div class="terminal-container" :style="{ height }">
    <div class="terminal-header">
      <span class="terminal-dot red"></span>
      <span class="terminal-dot yellow"></span>
      <span class="terminal-dot green"></span>
      <span class="terminal-title">{{ title }}</span>
    </div>
    <div class="terminal-body" ref="terminalRef">
      <div v-if="lines.length === 0" class="terminal-empty">
        <span class="prompt">$</span> Waiting for input...
      </div>
      <div v-for="(line, idx) in lines" :key="idx" class="terminal-line">
        <span v-if="line.type === 'cmd'" class="prompt">$ </span>
        <span v-else-if="line.type === 'success'" class="text-success">✓ </span>
        <span v-else-if="line.type === 'error'" class="text-error">✗ </span>
        <span v-else class="text-info">  </span>
        <span :class="'text-' + (line.type || 'info')">{{ line.text }}</span>
      </div>
      <div class="cursor-line">
        <span class="prompt">$</span>
        <span class="cursor">▊</span>
      </div>
    </div>
  </div>
</template>

<style scoped>
.terminal-container {
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  border: 1px solid #374151;
}

.terminal-header {
  background: #1f2937;
  padding: 8px 12px;
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
}

.terminal-dot {
  width: 10px;
  height: 10px;
  border-radius: 50%;
}
.terminal-dot.red { background: #ef4444; }
.terminal-dot.yellow { background: #f59e0b; }
.terminal-dot.green { background: #10b981; }

.terminal-title {
  margin-left: 8px;
  font-size: 12px;
  color: #9ca3af;
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
}

.terminal-body {
  background: #111827;
  flex: 1;
  overflow-y: auto;
  padding: 12px 16px;
  font-family: 'Menlo', 'Monaco', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.7;
}

.terminal-empty {
  color: #6b7280;
}

.terminal-line {
  white-space: pre-wrap;
  word-break: break-all;
}

.prompt {
  color: #10b981;
  font-weight: bold;
}

.text-cmd { color: #f9fafb; }
.text-info { color: #d1d5db; }
.text-success { color: #34d399; }
.text-error { color: #f87171; }
.text-warning { color: #fbbf24; }

.cursor-line {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-top: 2px;
}

.cursor {
  color: #10b981;
  animation: blink 1s step-end infinite;
}

@keyframes blink {
  0%, 100% { opacity: 1; }
  50% { opacity: 0; }
}

/* scrollbar */
.terminal-body::-webkit-scrollbar { width: 6px; }
.terminal-body::-webkit-scrollbar-track { background: #1f2937; }
.terminal-body::-webkit-scrollbar-thumb { background: #4b5563; border-radius: 3px; }
</style>
