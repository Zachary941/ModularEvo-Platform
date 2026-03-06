<script setup>
</script>

<template>
  <div class="chapter4-container">
    <el-page-header @back="$router.push('/')" title="返回首页">
      <template #content>
        <span class="page-title">第四章：GPT-Neo Router 动态合并</span>
      </template>
    </el-page-header>

    <!-- 知识图谱占位 -->
    <el-card class="section-card" shadow="never">
      <template #header>
        <span>知识图谱</span>
      </template>
      <div class="placeholder">
        <el-empty description="知识图谱组件 — 待 P5 实现" />
      </div>
    </el-card>

    <!-- 功能区 -->
    <el-row :gutter="16">
      <el-col :span="12">
        <el-card class="func-card" shadow="hover">
          <template #header>
            <span>预部署模型信息</span>
          </template>
          <el-empty description="待 P4.5 实现" :image-size="80" />
        </el-card>
      </el-col>
      <el-col :span="12">
        <el-card class="func-card" shadow="hover">
          <template #header>
            <span>数据集上传 & 评测</span>
          </template>
          <el-empty description="待 P4.6 实现" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>

    <!-- 评测结果占位 -->
    <el-card class="section-card" shadow="never" style="margin-top: 16px;">
      <template #header>
        <span>评测结果</span>
      </template>
      <el-empty description="待 P4.7 实现" :image-size="80" />
    </el-card>
  </div>
</template>

<style scoped>
.chapter4-container {
  max-width: 1200px;
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
.placeholder {
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
}
.func-card {
  min-height: 300px;
}
</style>
