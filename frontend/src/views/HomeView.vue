<script setup>
import { useRouter } from 'vue-router'

const router = useRouter()

const entries = [
  {
    emoji: '🧬',
    title: 'ModularEvo 进化',
    desc: '基于 CodeBERT 的模型模块化全流程：稀疏化训练 → 下游微调 → 多种合并方法对比评测。',
    tags: ['稀疏化', '微调', '合并评测'],
    route: '/modularevo',
    color: '#7c3aed',
  },
  {
    emoji: '🔀',
    title: 'AutoRouter 自动组合',
    desc: '基于 GPT-Neo 125M 的 Router 动态合并方法，支持上传混合数据集进行实时评测。',
    tags: ['Router', '动态合并', '实时评测'],
    route: '/autorouter',
    color: '#10b981',
  },
]
</script>

<template>
  <div class="home-view">
    <!-- Hero -->
    <div class="hero">
      <h1 class="hero-title">⚡ LLMOps Platform</h1>
      <p class="hero-sub">Transformer 模型模块化与动态合并 · 可视化展示系统</p>
    </div>

    <!-- Entry Cards -->
    <div class="cards-grid">
      <el-card
        v-for="item in entries"
        :key="item.route"
        class="entry-card"
        shadow="hover"
        @click="router.push(item.route)"
      >
        <div class="card-emoji">{{ item.emoji }}</div>
        <h2 class="card-title">{{ item.title }}</h2>
        <p class="card-desc">{{ item.desc }}</p>
        <div class="card-tags">
          <el-tag
            v-for="t in item.tags"
            :key="t"
            size="small"
            :style="{ borderColor: item.color, color: item.color }"
            effect="plain"
          >
            {{ t }}
          </el-tag>
        </div>
        <el-button :style="{ background: item.color, borderColor: item.color }" type="primary" round>
          进入 ▶
        </el-button>
      </el-card>
    </div>
  </div>
</template>

<style scoped>
.home-view {
  max-width: 900px;
  margin: 0 auto;
  padding: 48px 20px 40px;
}
.hero {
  text-align: center;
  margin-bottom: 48px;
}
.hero-title {
  font-size: 34px;
  font-weight: 700;
  color: var(--primary-color, #5b21b6);
  margin-bottom: 10px;
}
.hero-sub {
  font-size: 16px;
  color: var(--text-regular, #4b5563);
}
.cards-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
  gap: 28px;
}
.entry-card {
  text-align: center;
  padding: 32px 24px;
  cursor: pointer;
  transition: transform 0.2s;
  border-radius: 12px;
}
.entry-card:hover {
  transform: translateY(-6px);
}
.card-emoji {
  font-size: 48px;
  margin-bottom: 12px;
}
.card-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--text-primary, #1e1b4b);
  margin-bottom: 12px;
}
.card-desc {
  font-size: 14px;
  color: var(--text-regular, #4b5563);
  line-height: 1.8;
  margin-bottom: 18px;
}
.card-tags {
  display: flex;
  justify-content: center;
  gap: 8px;
  margin-bottom: 22px;
}
</style>
