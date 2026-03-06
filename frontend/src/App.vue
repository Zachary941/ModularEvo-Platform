<script setup>
</script>

<template>
  <el-container class="app-container">
    <el-header class="app-header">
      <div class="header-content">
        <router-link to="/" class="logo-link">
          <span class="logo-text">TransModular Demo</span>
        </router-link>
        <el-menu
          :default-active="$route.path"
          mode="horizontal"
          :router="true"
          class="nav-menu"
        >
          <el-menu-item index="/">首页</el-menu-item>
          <el-menu-item index="/chapter3">第三章：模块化</el-menu-item>
          <el-menu-item index="/chapter4">第四章：动态合并</el-menu-item>
        </el-menu>
      </div>
    </el-header>
    <el-main class="app-main">
      <router-view />
    </el-main>
  </el-container>
</template>

<style scoped>
.app-container {
  min-height: 100vh;
}
.app-header {
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
  padding: 0 20px;
  height: 60px;
  display: flex;
  align-items: center;
}
.header-content {
  display: flex;
  align-items: center;
  width: 100%;
}
.logo-link {
  text-decoration: none;
  margin-right: 40px;
}
.logo-text {
  font-size: 20px;
  font-weight: 700;
  color: #409eff;
}
.nav-menu {
  border-bottom: none;
}
.app-main {
  background: #f5f7fa;
  padding: 20px;
}
</style>
