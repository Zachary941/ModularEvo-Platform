<script setup>
import { computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { House, DataLine, SetUp, Share } from '@element-plus/icons-vue'

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
        <el-menu-item index="/models">
          <el-icon><DataLine /></el-icon>
          <span>模型介绍</span>
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
  background: var(--primary-color);
  display: flex;
  align-items: center;
  padding: 0 20px;
  box-shadow: 0 1px 4px rgba(0, 0, 0, 0.15);
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
  gap: 8px;
}

.logo-icon {
  font-size: 22px;
}

.logo-text {
  font-size: 18px;
  font-weight: 700;
  color: #ffffff;
  letter-spacing: 0.5px;
}

.header-center {
  flex: 1;
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
  padding: 0 18px;
}

.header-right {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success-color);
  box-shadow: 0 0 6px var(--success-color);
}

.status-text {
  font-size: 13px;
  color: rgba(255, 255, 255, 0.85);
}
</style>
