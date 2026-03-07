<script setup>
import AppHeader from './components/AppHeader.vue'
import AppSidebar from './components/AppSidebar.vue'
</script>

<template>
  <el-container class="app-layout" direction="vertical">
    <AppHeader />
    <el-container class="app-body">
      <AppSidebar />
      <el-main class="app-main">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<style scoped>
.app-layout {
  min-height: 100vh;
}
.app-body {
  padding-top: var(--header-height, 56px);
}
.app-main {
  margin-left: var(--sidebar-width, 220px);
  background: var(--bg-page, #f3f4f6);
  padding: 24px;
  min-height: calc(100vh - var(--header-height, 56px));
  overflow-y: auto;
}
</style>
