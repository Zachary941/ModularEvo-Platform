<script setup>
</script>

<template>
  <div class="models-view">
    <el-page-header title="返回首页" @back="$router.push('/')">
      <template #content>
        <span class="page-title">📦 模型介绍</span>
      </template>
    </el-page-header>
    <el-empty description="模型介绍页面 — 即将上线" :image-size="160" style="margin-top: 80px" />
  </div>
</template>

<style scoped>
.models-view {
  max-width: 1200px;
  margin: 0 auto;
}
.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
}
</style>
