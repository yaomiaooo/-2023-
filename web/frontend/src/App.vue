<template>
  <div id="app">
    <!-- 顶部导航 -->
    <header class="top-nav">
      <div class="nav-container">
        <div class="nav-brand">
          <div class="brand-icon">🧬</div>
          <span class="brand-text">医学文本挖掘系统</span>
        </div>
        <nav class="nav-menu">
          <router-link to="/" class="nav-link" :class="{ active: $route.path === '/' }">
            首页
          </router-link>
          <router-link to="/task1" class="nav-link" :class="{ active: $route.path === '/task1' }">
            实体抽取
          </router-link>
          <router-link to="/task2" class="nav-link" :class="{ active: $route.path === '/task2' }">
            文本分类
          </router-link>
          <router-link to="/stats" class="nav-link" :class="{ active: $route.path === '/stats' }">
            数据集统计
          </router-link>
        </nav>
        <div class="nav-actions">
          <el-button type="text" class="nav-btn" @click="showAbout = true">
            <InfoFilled />
            关于
          </el-button>
        </div>
      </div>
    </header>

    <!-- 主要内容 -->
    <main class="main-content">
      <router-view />
    </main>

    <!-- 背景装饰 -->
    <div class="bg-decoration">
      <div class="decoration-circle circle-1"></div>
      <div class="decoration-circle circle-2"></div>
      <div class="decoration-circle circle-3"></div>
      <div class="decoration-circle circle-4"></div>
      <div class="decoration-circle circle-5"></div>
    </div>

    <!-- 关于对话框 -->
    <el-dialog v-model="showAbout" title="关于系统" width="600px" class="about-dialog">
      <div class="about-content">
        <div class="about-header">
          <div class="about-icon">🧬</div>
          <h3>医学文本挖掘系统</h3>
        </div>
        <p class="about-desc">基于深度学习技术的智能医学文本分析平台</p>

        <div class="about-features">
          <div class="feature-item">
            <div class="feature-icon">🔬</div>
            <div class="feature-content">
              <h4>医学实体抽取</h4>
              <p>从医学文本中自动识别和提取实体，如疾病、症状、药物等9大类实体</p>
            </div>
          </div>
          <div class="feature-item">
            <div class="feature-icon">📋</div>
            <div class="feature-content">
              <h4>临床试验分类</h4>
              <p>对临床试验筛选标准进行智能分类，支持44种类别</p>
            </div>
          </div>
        </div>

        <div class="about-tech">
          <h4>技术栈</h4>
          <div class="tech-stack">
            <span class="tech-tag">Flask</span>
            <span class="tech-tag">PyTorch</span>
            <span class="tech-tag">Transformers</span>
            <span class="tech-tag">Vue 3</span>
            <span class="tech-tag">BERT</span>
            <span class="tech-tag">Global Pointer</span>
          </div>
        </div>
      </div>
      <template #footer>
        <el-button type="primary" @click="showAbout = false">知道了</el-button>
      </template>
    </el-dialog>
  </div>
</template>

<script setup>
import { ref } from 'vue'
import { InfoFilled } from '@element-plus/icons-vue'

const showAbout = ref(false)
</script>

<style>
/* 全局样式 */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
  line-height: 1.6;
  color: #065f46;
}

/* 顶部导航 */
.top-nav {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  z-index: 1000;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid rgba(16, 185, 129, 0.1);
  box-shadow: 0 2px 20px rgba(16, 185, 129, 0.1);
}

.nav-container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 20px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  height: 70px;
}

.nav-brand {
  display: flex;
  align-items: center;
  gap: 12px;
}

.brand-icon {
  font-size: 28px;
  animation: pulse 2s infinite;
}

.brand-text {
  font-size: 20px;
  font-weight: 700;
  color: #065f46;
  letter-spacing: -0.02em;
}

.nav-menu {
  display: flex;
  gap: 8px;
}

.nav-link {
  padding: 8px 16px;
  text-decoration: none;
  color: #065f46;
  font-weight: 500;
  border-radius: 8px;
  transition: all 0.3s ease;
  position: relative;
}

.nav-link:hover {
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
}

.nav-link.active {
  background: #10b981;
  color: white;
}

.nav-link.active::after {
  content: '';
  position: absolute;
  bottom: -1px;
  left: 50%;
  transform: translateX(-50%);
  width: 20px;
  height: 3px;
  background: #f59e0b;
  border-radius: 2px;
}

.nav-actions {
  display: flex;
  align-items: center;
}

.nav-btn {
  color: #065f46;
  font-weight: 500;
}

/* 主要内容 */
.main-content {
  padding-top: 70px;
  min-height: 100vh;
  background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 25%, #d1fae5 50%, #a7f3d0 75%, #6ee7b7 100%);
  position: relative;
  overflow-x: hidden;
}

/* 背景装饰 */
.bg-decoration {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  pointer-events: none;
  z-index: -1;
}

.decoration-circle {
  position: absolute;
  border-radius: 50%;
  background: radial-gradient(circle, rgba(16, 185, 129, 0.1) 0%, rgba(16, 185, 129, 0.05) 50%, transparent 70%);
  animation: float 6s ease-in-out infinite;
}

.circle-1 {
  width: 300px;
  height: 300px;
  top: 10%;
  left: -5%;
  animation-delay: 0s;
}

.circle-2 {
  width: 200px;
  height: 200px;
  top: 60%;
  right: -3%;
  animation-delay: 2s;
}

.circle-3 {
  width: 150px;
  height: 150px;
  bottom: 20%;
  left: 10%;
  animation-delay: 4s;
}

.circle-4 {
  width: 100px;
  height: 100px;
  top: 30%;
  right: 20%;
  animation-delay: 1s;
}

.circle-5 {
  width: 80px;
  height: 80px;
  bottom: 40%;
  left: 60%;
  animation-delay: 3s;
}

@keyframes float {
  0%, 100% {
    transform: translateY(0px);
  }
  50% {
    transform: translateY(-20px);
  }
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.05);
  }
}

/* 关于对话框 */
.about-dialog .el-dialog__header {
  margin: 0;
  padding: 24px 24px 0;
}

.about-dialog .el-dialog__body {
  padding: 16px 24px;
}

.about-dialog .el-dialog__footer {
  padding: 0 24px 24px;
}

.about-content {
  text-align: center;
}

.about-header {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 12px;
  margin-bottom: 16px;
}

.about-icon {
  font-size: 48px;
  animation: pulse 2s infinite;
}

.about-header h3 {
  color: #065f46;
  font-size: 24px;
  font-weight: 700;
}

.about-desc {
  color: #10b981;
  font-size: 16px;
  margin-bottom: 24px;
}

.about-features {
  display: grid;
  gap: 16px;
  margin-bottom: 24px;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 16px;
  background: rgba(16, 185, 129, 0.05);
  border-radius: 12px;
  border: 1px solid rgba(16, 185, 129, 0.1);
}

.feature-icon {
  font-size: 32px;
}

.feature-content {
  text-align: left;
}

.feature-content h4 {
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 4px;
}

.feature-content p {
  color: #10b981;
  font-size: 14px;
  margin: 0;
}

.about-tech h4 {
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 12px;
}

.tech-stack {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  justify-content: center;
}

.tech-tag {
  padding: 4px 12px;
  background: rgba(16, 185, 129, 0.1);
  color: #10b981;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 500;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .nav-container {
    padding: 0 16px;
    height: 60px;
  }

  .brand-text {
    display: none;
  }

  .nav-menu {
    display: none; /* 移动端可考虑添加汉堡菜单 */
  }

  .feature-item {
    flex-direction: column;
    text-align: center;
  }

  .feature-content {
    text-align: center;
  }
}
</style>
