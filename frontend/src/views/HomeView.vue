<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const entries = [
  {
    emoji: '🧬',
    title: 'ModularEvo 模块化进化',
    desc: '基于模块化的模型协同进化框架全流程：通过模块化训练提取功能模块，结合下游微调与多种合并方法（Task Arithmetic、TIES、DARE、简单平均），实现跨任务知识迁移与模型能力组合。',
    tags: ['稀疏化训练', '模块提取', '微调适配', '合并评测'],
    route: '/modularevo',
    color: '#7c3aed',
    gradient: 'linear-gradient(135deg, #7c3aed 0%, #a855f7 50%, #c084fc 100%)',
    image: '/images/modularevo.png',
    highlights: ['CodeBERT 预训练模型', '4 种合并策略对比', '知识图谱可视化'],
  },
  {
    emoji: '🔀',
    title: 'AutoRouter 自动组合',
    desc: '基于输入识别的 Router 动态合并方法：训练路由器权重，智能选择最优模块组合，支持上传混合数据集进行实时评测，展示归一化准确率。',
    tags: ['Router 路由', '动态合并', 'Task Vector', '实时评测'],
    route: '/autorouter',
    color: '#10b981',
    gradient: 'linear-gradient(135deg, #10b981 0%, #34d399 50%, #6ee7b7 100%)',
    image: '/images/router.png',
    highlights: ['GPT-Neo 125M 模型', 'Router 动态权重', '自动匹配分类头'],
  },
]
</script>

<template>
  <div class="home-view">
    <!-- Hero Section -->
    <div class="hero">
      <div class="hero-bg-decor">
        <div class="decor-circle c1"></div>
        <div class="decor-circle c2"></div>
        <div class="decor-circle c3"></div>
      </div>
      <div class="hero-content">
        <div class="hero-badge">🚀 Transformer Model Modularization</div>
        <h1 class="hero-title">
          <span class="title-gradient">LLMOps</span> Platform
        </h1>
        <p class="hero-sub">Transformer 模型模块化与动态合并 · 可视化演示系统</p>
        <div class="hero-stats">
          <div class="stat-item">
            <span class="stat-value">2</span>
            <span class="stat-label">核心算法</span>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <span class="stat-value">4+</span>
            <span class="stat-label">合并策略</span>
          </div>
          <div class="stat-divider"></div>
          <div class="stat-item">
            <span class="stat-value">实时</span>
            <span class="stat-label">可视化评测</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Algorithm Entry Cards -->
    <div class="section-title">
      <span class="section-icon">📋</span>
      <span>算法模块</span>
    </div>

    <div class="cards-list">
      <div
        v-for="(item, idx) in entries"
        :key="item.route"
        class="algo-card"
        :class="{ 'card-reverse': idx % 2 === 1 }"
        @click="router.push(item.route)"
      >
        <!-- Image Side -->
        <div class="card-image-wrapper">
          <div class="image-border-glow" :style="{ background: item.gradient }"></div>
          <div class="card-image-container">
            <img :src="item.image" :alt="item.title" class="card-image" />
          </div>
        </div>

        <!-- Info Side -->
        <div class="card-info">
          <div class="card-header">
            <span class="card-emoji">{{ item.emoji }}</span>
            <h2 class="card-title">{{ item.title }}</h2>
          </div>
          <p class="card-desc">{{ item.desc }}</p>

          <!-- Highlights -->
          <div class="card-highlights">
            <div v-for="h in item.highlights" :key="h" class="highlight-item">
              <span class="highlight-dot" :style="{ background: item.color }"></span>
              <span>{{ h }}</span>
            </div>
          </div>

          <!-- Tags -->
          <div class="card-tags">
            <el-tag
              v-for="t in item.tags"
              :key="t"
              size="small"
              :style="{ borderColor: item.color + '60', color: item.color, background: item.color + '10' }"
              effect="plain"
              round
            >
              {{ t }}
            </el-tag>
          </div>

          <!-- Action -->
          <el-button
            :style="{ background: item.gradient, borderColor: 'transparent', boxShadow: '0 4px 14px ' + item.color + '40' }"
            type="primary"
            round
            size="large"
          >
            进入演示 ▶
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.home-view {
  max-width: 1060px;
  margin: 0 auto;
  padding: 36px 28px 60px;
}

/* ---- Hero ---- */
.hero {
  position: relative;
  text-align: center;
  padding: 52px 24px 40px;
  margin-bottom: 44px;
  border-radius: 20px;
  background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 40%, #e0e7ff 100%);
  overflow: hidden;
}
.hero-bg-decor {
  position: absolute;
  inset: 0;
  pointer-events: none;
}
.decor-circle {
  position: absolute;
  border-radius: 50%;
  opacity: 0.25;
}
.c1 {
  width: 200px; height: 200px;
  background: radial-gradient(circle, #7c3aed, transparent 70%);
  top: -60px; right: -40px;
}
.c2 {
  width: 140px; height: 140px;
  background: radial-gradient(circle, #10b981, transparent 70%);
  bottom: -30px; left: -20px;
}
.c3 {
  width: 90px; height: 90px;
  background: radial-gradient(circle, #6366f1, transparent 70%);
  top: 40%; left: 60%;
}
.hero-content {
  position: relative;
  z-index: 1;
}
.hero-badge {
  display: inline-block;
  padding: 5px 16px;
  border-radius: 20px;
  background: rgba(124, 58, 237, 0.1);
  color: #7c3aed;
  font-size: 13px;
  font-weight: 500;
  margin-bottom: 18px;
  letter-spacing: 0.3px;
}
.hero-title {
  font-size: 38px;
  font-weight: 800;
  color: var(--text-primary, #1e1b4b);
  margin-bottom: 12px;
  letter-spacing: -0.5px;
}
.title-gradient {
  background: linear-gradient(135deg, #7c3aed, #6366f1, #10b981);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-size: 16px;
  color: var(--text-regular, #4b5563);
  margin-bottom: 28px;
}
.hero-stats {
  display: inline-flex;
  align-items: center;
  gap: 24px;
  padding: 14px 32px;
  background: rgba(255,255,255,0.75);
  backdrop-filter: blur(8px);
  border-radius: 14px;
  border: 1px solid rgba(124, 58, 237, 0.1);
}
.stat-item {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 2px;
}
.stat-value {
  font-size: 22px;
  font-weight: 700;
  color: var(--primary-color, #5b21b6);
}
.stat-label {
  font-size: 12px;
  color: var(--text-secondary, #9ca3af);
}
.stat-divider {
  width: 1px;
  height: 32px;
  background: var(--border-color, #e5e7eb);
}

/* ---- Section Title ---- */
.section-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 20px;
  font-weight: 700;
  color: var(--text-primary, #1e1b4b);
  margin-bottom: 24px;
}
.section-icon {
  font-size: 22px;
}

/* ---- Algorithm Cards ---- */
.cards-list {
  display: flex;
  flex-direction: column;
  gap: 28px;
}
.algo-card {
  display: flex;
  gap: 32px;
  padding: 28px;
  background: var(--bg-card, #fff);
  border-radius: 16px;
  border: 1px solid var(--border-color, #e5e7eb);
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.algo-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(91, 33, 182, 0.1);
  border-color: rgba(124, 58, 237, 0.2);
}
.algo-card.card-reverse {
  flex-direction: row-reverse;
}

/* Image Side */
.card-image-wrapper {
  position: relative;
  flex-shrink: 0;
  width: 380px;
  align-self: stretch;
}
.image-border-glow {
  position: absolute;
  inset: -2px;
  border-radius: 14px;
  opacity: 0.6;
  z-index: 0;
  height: calc(100% + 4px);
}
.card-image-container {
  position: relative;
  z-index: 1;
  border-radius: 12px;
  overflow: hidden;
  background: #fff;
  border: 2px solid rgba(255,255,255,0.8);
  height: 100%;
}
.card-image {
  width: 100%;
  height: 100%;
  object-fit: contain;
  display: block;
  transition: transform 0.4s ease;
  background: #fff;
}
.algo-card:hover .card-image {
  transform: scale(1.03);
}

/* Info Side */
.card-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
  min-width: 0;
}
.card-header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 14px;
}
.card-emoji {
  font-size: 32px;
}
.card-title {
  font-size: 22px;
  font-weight: 700;
  color: var(--text-primary, #1e1b4b);
}
.card-desc {
  font-size: 14px;
  color: var(--text-regular, #4b5563);
  line-height: 1.85;
  margin-bottom: 16px;
}
.card-highlights {
  display: flex;
  flex-direction: column;
  gap: 6px;
  margin-bottom: 16px;
}
.highlight-item {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--text-regular, #4b5563);
}
.highlight-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}
.card-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 20px;
}
</style>
