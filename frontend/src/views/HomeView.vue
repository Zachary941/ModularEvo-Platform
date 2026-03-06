<script setup>
import { useRouter } from 'vue-router'
import { Monitor, DataAnalysis } from '@element-plus/icons-vue'

const router = useRouter()
</script>

<template>
  <div class="home-container">
    <!-- 系统简介 -->
    <div class="hero-section">
      <h1 class="hero-title">TransModular Demo</h1>
      <p class="hero-subtitle">
        Transformer 模型模块化与动态合并展示系统
      </p>
      <p class="hero-desc">
        本系统展示了基于 Transformer 的模型模块化方法与动态合并算法。
        涵盖第三章（CodeBERT 模块化全流程）与第四章（GPT-Neo Router 动态合并）的核心研究成果。
      </p>
    </div>

    <!-- 章节入口卡片 -->
    <div class="cards-section">
      <el-row :gutter="24" justify="center">
        <el-col :xs="24" :sm="12" :md="10">
          <el-card class="chapter-card" shadow="hover" @click="router.push('/chapter3')">
            <div class="card-icon">
              <el-icon :size="48" color="#409eff"><Monitor /></el-icon>
            </div>
            <h2 class="card-title">第三章：CodeBERT 模块化</h2>
            <p class="card-desc">
              基于 CodeBERT 的模型模块化全流程展示，包含模块化训练、下游任务微调与多种模型合并方法对比。
            </p>
            <div class="card-flow">
              <el-tag>模块化</el-tag>
              <span class="arrow">→</span>
              <el-tag type="success">微调</el-tag>
              <span class="arrow">→</span>
              <el-tag type="warning">合并</el-tag>
            </div>
            <el-button type="primary" class="card-btn">进入 ▶</el-button>
          </el-card>
        </el-col>

        <el-col :xs="24" :sm="12" :md="10">
          <el-card class="chapter-card" shadow="hover" @click="router.push('/chapter4')">
            <div class="card-icon">
              <el-icon :size="48" color="#67c23a"><DataAnalysis /></el-icon>
            </div>
            <h2 class="card-title">第四章：GPT-Neo Router 动态合并</h2>
            <p class="card-desc">
              基于 GPT-Neo 125M 的 Router 动态合并方法展示，支持上传混合数据集进行实时评测。
            </p>
            <div class="card-flow">
              <el-tag>预部署模型</el-tag>
              <span class="arrow">→</span>
              <el-tag type="success">上传数据</el-tag>
              <span class="arrow">→</span>
              <el-tag type="warning">评测</el-tag>
            </div>
            <el-button type="success" class="card-btn">进入 ▶</el-button>
          </el-card>
        </el-col>
      </el-row>
    </div>
  </div>
</template>

<style scoped>
.home-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
}
.hero-section {
  text-align: center;
  margin-bottom: 48px;
}
.hero-title {
  font-size: 36px;
  font-weight: 700;
  color: #303133;
  margin-bottom: 8px;
}
.hero-subtitle {
  font-size: 18px;
  color: #606266;
  margin-bottom: 16px;
}
.hero-desc {
  font-size: 14px;
  color: #909399;
  max-width: 640px;
  margin: 0 auto;
  line-height: 1.8;
}
.cards-section {
  margin-top: 20px;
}
.chapter-card {
  text-align: center;
  padding: 24px 16px;
  cursor: pointer;
  transition: transform 0.2s;
  min-height: 320px;
  display: flex;
  flex-direction: column;
  align-items: center;
}
.chapter-card:hover {
  transform: translateY(-4px);
}
.card-icon {
  margin-bottom: 16px;
}
.card-title {
  font-size: 20px;
  font-weight: 600;
  color: #303133;
  margin-bottom: 12px;
}
.card-desc {
  font-size: 14px;
  color: #606266;
  line-height: 1.7;
  margin-bottom: 16px;
}
.card-flow {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-bottom: 20px;
}
.arrow {
  color: #c0c4cc;
  font-size: 16px;
}
.card-btn {
  margin-top: auto;
}
</style>
