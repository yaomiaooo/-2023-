<template>
    <div class="home-view">
      <!-- 页面上层：主题聚焦区域 -->
      <section class="hero-section">
        <div class="hero-container">
          <div class="hero-content">
            <h1 class="hero-title">
              智能医学文本分析
              <span class="hero-accent">平台</span>
            </h1>
            <p class="hero-description">
              基于深度学习技术的医学文本挖掘系统，为医疗科研提供智能分析工具
            </p>
  
            <div class="hero-features">
              <div class="feature-item">
                <div class="feature-icon">🌱</div>
                <div class="feature-text">
                  <h3>精准识别</h3>
                  <p>自动提取医学实体</p>
                </div>
              </div>
              <div class="feature-item">
                <div class="feature-icon">🧬</div>
                <div class="feature-text">
                  <h3>智能分类</h3>
                  <p>临床试验标准分类</p>
                </div>
              </div>
              <div class="feature-item">
                <div class="feature-icon">🦋</div>
                <div class="feature-text">
                  <h3>高效分析</h3>
                  <p>快速处理大量文本</p>
                </div>
              </div>
            </div>
  
            <div class="hero-actions">
              <router-link to="/task1" class="btn-primary">
                开始实体抽取
                <ArrowRight />
              </router-link>
              <router-link to="/task2" class="btn-secondary">
                尝试文本分类
              </router-link>
            </div>
          </div>
  
        </div>
      </section>
  
      <!-- 页面下层：功能卡片区域 -->
      <section class="features-section">
        <div class="container">
          <div class="section-header">
            <h2 class="section-title">核心功能</h2>
            <p class="section-description">
              强大的AI能力，为医学研究提供精准、高效的文本分析服务
            </p>
          </div>
  
          <div class="features-grid">
            <!-- 功能卡片1：医学实体抽取 -->
            <div class="feature-card">
              <div class="card-decoration"></div>
              <div class="card-header">
                <div class="card-icon">🧠</div>
                <h3 class="card-title">医学实体抽取</h3>
              </div>
              <div class="card-content">
                <p class="card-description">
                  从医学文本中自动识别和提取9类关键实体，包括疾病、症状、药物、医疗设备等。
                </p>
                <ul class="card-features">
                  <li>🔬 疾病识别</li>
                  <li>💊 药物提取</li>
                  <li>🏥 医疗设备</li>
                  <li>📊 检验项目</li>
                </ul>
              </div>
              <div class="card-footer">
                <router-link to="/task1" class="card-btn">
                  开始使用
                  <ArrowRight />
                </router-link>
              </div>
            </div>
  
            <!-- 功能卡片2：文本分类 -->
            <div class="feature-card">
              <div class="card-decoration"></div>
              <div class="card-header">
                <div class="card-icon">🧬</div>
                <h3 class="card-title">临床试验分类</h3>
              </div>
              <div class="card-content">
                <p class="card-description">
                  对临床试验筛选标准进行智能分类，支持44种不同的筛选标准类别。
                </p>
                <ul class="card-features">
                  <li>📋 标准分类</li>
                  <li>🎯 精准匹配</li>
                  <li>⚡ 快速处理</li>
                  <li>📈 统计分析</li>
                </ul>
              </div>
              <div class="card-footer">
                <router-link to="/task2" class="card-btn">
                  开始使用
                  <ArrowRight />
                </router-link>
              </div>
            </div>
  
            <!-- 功能卡片3：技术特性
            <div class="feature-card">
              <div class="card-decoration"></div>
              <div class="card-header">
                <div class="card-icon">🚀</div>
                <h3 class="card-title">AI技术驱动</h3>
              </div>
              <div class="card-content">
                <p class="card-description">
                  基于最先进的深度学习模型，提供高精度、高效率的文本分析服务。
                </p>
                <ul class="card-features">
                  <li>🤖 BERT模型</li>
                  <li>🎯 全局指针网络</li>
                  <li>⚡ 实时处理</li>
                  <li>🔒 隐私保护</li>
                </ul>
              </div>
              <div class="card-footer">
                <el-button type="text" class="card-btn-text" @click="showTechDialog = true">
                  了解技术
                </el-button>
              </div>
            </div> -->
          </div>
        </div>
      </section>
  
      <!-- 技术详情对话框 -->
      <el-dialog v-model="showTechDialog" title="技术架构" width="700px" class="tech-dialog">
        <div class="tech-content">
          <div class="tech-overview">
            <h3>🏗️ 系统架构</h3>
            <div class="tech-diagram">
              <div class="tech-layer frontend">
                <div class="layer-icon">💻</div>
                <div class="layer-info">
                  <h4>前端界面</h4>
                  <p>Vue 3 + Element Plus  </p>
                </div>
              </div>
              <div class="tech-arrow">⬇️</div>
              <div class="tech-layer backend">
                <div class="layer-icon">🔧</div>
                <div class="layer-info">
                  <h4>后端服务</h4>
                  <p>Flask + RESTful API</p>
                </div>
              </div>
              <div class="tech-arrow">⬇️</div>
              <div class="tech-layer ai">
                <div class="layer-icon">🧠</div>
                <div class="layer-info">
                  <h4>AI模型</h4>
                  <p>PyTorch + Transformers + BERT</p>
                </div>
              </div>
            </div>
          </div>
  
          <div class="tech-models">
            <h3>🎯 AI模型详解</h3>
            <div class="model-cards">
              <div class="model-card">
                <div class="model-icon">🔬</div>
                <h4>实体抽取模型</h4>
                <p>基于Global Pointer网络的嵌套实体识别模型</p>
                <ul>
                  <li>支持嵌套实体识别</li>
                  <li>9类医学实体</li>
                  <li>滑动窗口处理长文本</li>
                </ul>
              </div>
              <div class="model-card">
                <div class="model-icon">📋</div>
                <h4>分类模型</h4>
                <p>基于BERT的文本分类模型</p>
                <ul>
                  <li>44种筛选标准类别</li>
                  <li>多标签分类</li>
                  <li>置信度输出</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
        <template #footer>
          <el-button @click="showTechDialog = false">关闭</el-button>
        </template>
      </el-dialog>
    </div>
  </template>
  
  <script setup>
  import { ref } from 'vue'
  import { ArrowRight } from '@element-plus/icons-vue'
  
  const showTechDialog = ref(false)
  </script>
  
  <style scoped>
  .home-view {
    min-height: 100vh;
  }
  
  /* Hero Section */
  .hero-section {
    padding: 80px 0 60px;
    position: relative;
  }
  
  .hero-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 60px;
    align-items: center;
  }
  
  .hero-content {
    animation: fadeInUp 1s ease-out;
  }
  
  .hero-title {
    font-size: 48px;
    font-weight: 800;
    color: #065f46;
    line-height: 1.2;
    margin-bottom: 16px;
    letter-spacing: -0.02em;
  }
  
  .hero-accent {
    color: #10b981;
    position: relative;
  }
  
  .hero-accent::after {
    content: '';
    position: absolute;
    bottom: -4px;
    left: 0;
    width: 100%;
    height: 4px;
    background: linear-gradient(90deg, #f59e0b, #10b981);
    border-radius: 2px;
  }
  
  .hero-description {
    font-size: 20px;
    color: #10b981;
    line-height: 1.6;
    margin-bottom: 40px;
  }
  
  .hero-features {
    display: flex;
    gap: 24px;
    margin-bottom: 40px;
  }
  
  .feature-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 16px 20px;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 16px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
  }
  
  .feature-item:hover {
    transform: translateY(-4px);
    box-shadow: 0 8px 25px rgba(16, 185, 129, 0.15);
  }
  
  .feature-icon {
    font-size: 32px;
    animation: bounce 2s infinite;
  }
  
  .feature-text h3 {
    font-size: 16px;
    font-weight: 600;
    color: #065f46;
    margin-bottom: 4px;
  }
  
  .feature-text p {
    font-size: 14px;
    color: #10b981;
    margin: 0;
  }
  
  .hero-actions {
    display: flex;
    gap: 16px;
    align-items: center;
  }
  
  .btn-primary, .btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 14px 24px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
  }
  
  .btn-primary {
    background: linear-gradient(135deg, #10b981, #065f46);
    color: white;
  }
  
  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
  }
  
  .btn-secondary {
    background: rgba(255, 255, 255, 0.9);
    color: #065f46;
    border: 2px solid #10b981;
  }
  
  .btn-secondary:hover {
    background: #10b981;
    color: white;
    transform: translateY(-2px);
  }
  

  
  .stat-item {
    padding: 16px;
    background: rgba(255, 255, 255, 0.8);
    border-radius: 12px;
    backdrop-filter: blur(10px);
  }
  
  .stat-item:nth-child(3) {
    grid-column: span 2;
  }
  
  .stat-number {
    font-size: 32px;
    font-weight: 800;
    color: #065f46;
    line-height: 1;
  }
  
  .stat-label {
    font-size: 12px;
    color: #10b981;
    font-weight: 500;
    margin-top: 4px;
  }
  
  /* Features Section */
  .features-section {
    padding: 80px 0;
    background: rgba(255, 255, 255, 0.5);
    backdrop-filter: blur(10px);
  }
  
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 20px;
  }
  
  .section-header {
    text-align: center;
    margin-bottom: 60px;
  }
  
  .section-title {
    font-size: 36px;
    font-weight: 700;
    color: #065f46;
    margin-bottom: 16px;
    position: relative;
  }
  
  .section-title::after {
    content: '';
    position: absolute;
    bottom: -8px;
    left: 50%;
    transform: translateX(-50%);
    width: 60px;
    height: 4px;
    background: linear-gradient(90deg, #10b981, #34d399);
    border-radius: 2px;
  }
  
  .section-description {
    font-size: 18px;
    color: #10b981;
    max-width: 600px;
    margin: 0 auto;
  }
  
  .features-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
    gap: 40px;
  }
  
  .feature-card {
    background: rgba(255, 255, 255, 0.9);
    border-radius: 20px;
    padding: 32px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    backdrop-filter: blur(10px);
  }
  
  .feature-card:hover {
    transform: translateY(-8px);
    box-shadow: 0 20px 40px rgba(16, 185, 129, 0.15);
    border-color: rgba(16, 185, 129, 0.4);
  }
  
  .card-decoration {
    position: absolute;
    top: 0;
    right: 0;
    width: 80px;
    height: 80px;
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(16, 185, 129, 0.1));
    border-radius: 0 20px 0 50px;
  }
  
  .card-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
  }
  
  .card-icon {
    font-size: 48px;
    width: 80px;
    height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(135deg, #10b981, #34d399);
    border-radius: 16px;
    color: white;
  }
  
  .card-title {
    font-size: 24px;
    font-weight: 700;
    color: #065f46;
    margin: 0;
  }
  
  .card-content {
    margin-bottom: 24px;
  }
  
  .card-description {
    color: #10b981;
    font-size: 16px;
    line-height: 1.6;
    margin-bottom: 16px;
  }
  
  .card-features {
    list-style: none;
    padding: 0;
    margin: 0;
    display: grid;
    gap: 8px;
  }
  
  .card-features li {
    color: #065f46;
    font-size: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .card-footer {
    text-align: center;
  }
  
  .card-btn, .card-btn-text {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 12px 24px;
    border-radius: 12px;
    text-decoration: none;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    background: linear-gradient(135deg, #10b981, #065f46);
    color: white;
    border: none;
    cursor: pointer;
  }
  
  .card-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
  }
  
  .card-btn-text {
    background: transparent;
    color: #10b981;
    border: 2px solid #10b981;
  }
  
  .card-btn-text:hover {
    background: #10b981;
    color: white;
  }
  
  /* 技术对话框 */
  .tech-dialog .el-dialog__header {
    margin: 0;
    padding: 24px 24px 0;
  }
  
  .tech-dialog .el-dialog__body {
    padding: 16px 24px;
  }
  
  .tech-content {
    text-align: left;
  }
  
  .tech-overview h3, .tech-models h3 {
    color: #065f46;
    font-size: 20px;
    font-weight: 600;
    margin-bottom: 20px;
  }
  
  .tech-diagram {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 16px;
    margin-bottom: 40px;
  }
  
  .tech-layer {
    display: flex;
    align-items: center;
    gap: 16px;
    padding: 20px 24px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.2);
    min-width: 280px;
  }
  
  .layer-icon {
    font-size: 32px;
  }
  
  .layer-info h4 {
    font-size: 16px;
    font-weight: 600;
    color: #065f46;
    margin-bottom: 4px;
  }
  
  .layer-info p {
    font-size: 14px;
    color: #10b981;
    margin: 0;
  }
  
  .tech-arrow {
    font-size: 24px;
    color: #10b981;
  }
  
  .model-cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
  }
  
  .model-card {
    padding: 24px;
    background: rgba(255, 255, 255, 0.9);
    border-radius: 12px;
    border: 1px solid rgba(16, 185, 129, 0.2);
  }
  
  .model-icon {
    font-size: 32px;
    margin-bottom: 12px;
  }
  
  .model-card h4 {
    font-size: 18px;
    font-weight: 600;
    color: #065f46;
    margin-bottom: 8px;
  }
  
  .model-card > p {
    color: #10b981;
    font-size: 14px;
    margin-bottom: 16px;
  }
  
  .model-card ul {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  
  .model-card li {
    color: #065f46;
    font-size: 14px;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  
  .model-card li:before {
    content: '✓';
    color: #10b981;
    font-weight: bold;
  }
  
  /* 动画 */
  @keyframes fadeInUp {
    from {
      opacity: 0;
      transform: translateY(30px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
  
  @keyframes fadeInRight {
    from {
      opacity: 0;
      transform: translateX(30px);
    }
    to {
      opacity: 1;
      transform: translateX(0);
    }
  }
  
  @keyframes bounce {
    0%, 20%, 50%, 80%, 100% {
      transform: translateY(0);
    }
    40% {
      transform: translateY(-10px);
    }
    60% {
      transform: translateY(-5px);
    }
  }
  
  @keyframes rotate {
    from {
      transform: rotate(0deg);
    }
    to {
      transform: rotate(360deg);
    }
  }
  
  /* 响应式设计 */
  @media (max-width: 1024px) {
    .hero-container {
      grid-template-columns: 1fr;
      gap: 40px;
      text-align: center;
    }
  
    .hero-features {
      justify-content: center;
    }
  
    .hero-actions {
      justify-content: center;
    }
  
    .features-grid {
      grid-template-columns: 1fr;
    }
  }
  
  @media (max-width: 768px) {
    .hero-section {
      padding: 60px 0 40px;
    }
  
    .hero-title {
      font-size: 36px;
    }
  
    .hero-description {
      font-size: 18px;
    }
  
    .hero-features {
      flex-direction: column;
      align-items: center;
    }
  
    .hero-actions {
      flex-direction: column;
    }
  
    .btn-primary, .btn-secondary {
      width: 100%;
      justify-content: center;
    }
  
    .visual-circle {
      width: 220px;
      height: 220px;
    }
  
    .stat-number {
      font-size: 24px;
    }
  
    .features-section {
      padding: 60px 0;
    }
  
    .section-title {
      font-size: 28px;
    }
  
    .feature-card {
      padding: 24px;
    }
  
    .card-icon {
      width: 60px;
      height: 60px;
      font-size: 36px;
    }
  
    .card-title {
      font-size: 20px;
    }
  }
  </style>