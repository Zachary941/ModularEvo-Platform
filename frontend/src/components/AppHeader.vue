<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { House, SetUp, Share } from '@element-plus/icons-vue'

const route = useRoute()
const router = useRouter()

const activeIndex = computed(() => route.path)

const handleSelect = (index) => {
  router.push(index)
}
</script>

<template>
  <div class="app-header">
    <div class="header-left">
      <router-link to="/" class="logo-link">
        <span class="logo-icon">⚡</span>
        <span class="logo-text">LLMOps Platform</span>
      </router-link>
    </div>

    <div class="header-center">
      <el-menu
        :default-active="activeIndex"
        mode="horizontal"
        class="header-menu"
        @select="handleSelect"
        :ellipsis="false"
      >
        <el-menu-item index="/">
          <el-icon><House /></el-icon>
          <span>首页</span>
        </el-menu-item>
        <el-menu-item index="/modularevo">
          <el-icon><SetUp /></el-icon>
          <span>ModularEvo 进化</span>
        </el-menu-item>
        <el-menu-item index="/autorouter">
          <el-icon><Share /></el-icon>
          <span>AutoRouter 自动组合</span>
        </el-menu-item>
      </el-menu>
    </div>

    <div class="header-right">
      <span class="status-dot"></span>
      <span class="status-text">系统正常</span>
    </div>
  </div>
</template>

<style scoped>
.app-header {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  height: var(--header-height);
  background: linear-gradient(135deg, var(--primary-dark, #4c1d95) 0%, var(--primary-color, #5b21b6) 50%, var(--primary-light, #7c3aed) 100%);
  display: flex;
  align-items: center;
  padding: 0 24px;
  box-shadow: 0 2px 12px rgba(91, 33, 182, 0.3);
  z-index: 100;
}

.header-left {
  flex-shrink: 0;
  margin-right: 32px;
}

.logo-link {
  display: flex;
  align-items: center;
  text-decoration: none;
  gap: 10px;
  transition: opacity 0.2s;
}
.logo-link:hover {
  opacity: 0.9;
}

.logo-icon {
  font-size: 24px;
  filter: drop-shadow(0 0 4px rgba(255,255,255,0.3));
}

.logo-text {
  font-size: 18px;
  font-weight: 800;
  color: #ffffff;
  letter-spacing: 0.5px;
  text-shadow: 0 1px 2px rgba(0,0,0,0.1);
}

.header-center {
  flex: 1;
  display: flex;
  justify-content: center;
}

.header-menu {
  background: transparent !important;
  border-bottom: none !important;
  height: var(--header-height);
}

.header-menu .el-menu-item {
  height: var(--header-height);
  line-height: var(--header-height);
  font-size: 14px;
  font-weight: 500;
  padding: 0 20px;
  border-radius: 8px;
  margin: 0 2px;
  transition: all 0.2s ease !important;
}

.header-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 4px 14px;
  border-radius: 20px;
  background: rgba(255,255,255,0.1);
  backdrop-filter: blur(4px);
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success-color);
  box-shadow: 0 0 8px var(--success-color);
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 4px var(--success-color); }
  50% { box-shadow: 0 0 12px var(--success-color); }
}

.status-text {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.9);
  font-weight: 500;
}
</style>
