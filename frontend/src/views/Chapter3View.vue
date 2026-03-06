<script setup>
</script>

<template>
  <div class="chapter3-container">
    <el-page-header @back="$router.push('/')" title="返回首页">
      <template #content>
        <span class="page-title">第三章：CodeBERT 模块化全流程</span>
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

    <!-- 三步流程 -->
    <el-row :gutter="16">
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag>Step 1</el-tag>
              <span>模块化 (Modularization)</span>
            </div>
          </template>
          <el-empty description="待 P2.5 实现" :image-size="80" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag type="success">Step 2</el-tag>
              <span>下游任务微调</span>
            </div>
          </template>
          <el-empty description="待 P2.6 实现" :image-size="80" />
        </el-card>
      </el-col>
      <el-col :span="8">
        <el-card class="step-card" shadow="hover">
          <template #header>
            <div class="step-header">
              <el-tag type="warning">Step 3</el-tag>
              <span>模型合并</span>
            </div>
          </template>
          <el-empty description="待 P2.7 实现" :image-size="80" />
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<style scoped>
.chapter3-container {
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
.step-card {
  min-height: 300px;
}
.step-header {
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
