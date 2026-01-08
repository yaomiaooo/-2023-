<template>
  <div class="task2-container">
    <div class="header-section">
      <h2>📋 临床试验筛选标准分类</h2>
      <p class="description">
        对临床试验筛选标准进行自动分类，系统支持44种不同的筛选标准类别。
      </p>
    </div>

    <div class="main-content">
      <!-- 输入区域 -->
      <div class="input-section">
        <div class="input-header">
          <h3>输入筛选标准文本</h3>
          <el-button type="primary" @click="loadExample" size="small">
            加载示例
          </el-button>
        </div>

        <el-input
          v-model="inputText"
          type="textarea"
          :rows="4"
          placeholder="请输入临床试验筛选标准文本..."
          :disabled="loading"
        />

        <div class="button-group">
          <el-button
            type="primary"
            :loading="loading"
            @click="predictCategory"
            :disabled="!inputText.trim()"
          >
            开始分类
          </el-button>
          <el-button @click="clearAll">清空</el-button>
        </div>
      </div>

      <!-- 结果区域 -->
      <div v-if="results" class="result-section">
        <div class="result-header">
          <h3>分类结果</h3>
        </div>

        <!-- 主要预测结果 -->
        <div class="main-prediction">
          <div class="prediction-card">
            <div class="prediction-label">预测类别</div>
            <div class="prediction-value">{{ results.prediction }}</div>
          </div>
        </div>

        <!-- 概率分布 -->
        <div class="probability-section">
          <h4>概率分布（Top 5）</h4>
          <div class="probability-list">
            <div
              v-for="(item, index) in results.top_probabilities"
              :key="index"
              class="probability-item"
            >
              <div class="probability-info">
                <span class="category-name">{{ item.class }}</span>
                <span class="probability-value">{{ (item.probability * 100).toFixed(2) }}%</span>
              </div>
              <el-progress
                :percentage="item.probability * 100"
                :color="getProgressColor(index)"
                :show-text="false"
                :stroke-width="8"
              />
            </div>
          </div>
        </div>

        <!-- 输入文本回顾 -->
        <div class="text-review">
          <h4>输入文本</h4>
          <div class="review-text">{{ results.text }}</div>
        </div>
      </div>

      <!-- 错误信息 -->
      <div v-if="error" class="error-section">
        <el-alert
          :title="error"
          type="error"
          :closable="false"
          show-icon
        />
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 响应式数据
const inputText = ref('')
const loading = ref(false)
const results = ref(null)
const error = ref('')
const examples = ref([])

// 加载示例数据
const loadExample = async () => {
  try {
    const response = await axios.get('/api/task2/examples')
    examples.value = response.data.examples
    if (examples.value.length > 0) {
      inputText.value = examples.value[Math.floor(Math.random() * examples.value.length)]
    }
  } catch (err) {
    ElMessage.error('加载示例失败')
  }
}

// 执行分类预测
const predictCategory = async () => {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入文本')
    return
  }

  loading.value = true
  error.value = ''
  results.value = null

  try {
    const response = await axios.post('/api/task2/predict', {
      text: inputText.value
    })

    results.value = response.data
    ElMessage({
      message: '分类完成',
      type: 'success',
      duration: 1500
    })
  } catch (err) {
    error.value = err.response?.data?.error || '分类失败，请重试'
    ElMessage.error(error.value)
  } finally {
    loading.value = false
  }
}

// 清空所有内容
const clearAll = () => {
  inputText.value = ''
  results.value = null
  error.value = ''
}

// 获取进度条颜色
const getProgressColor = (index) => {
  const colors = [
    '#409eff', // 蓝色 - 第一名
    '#67c23a', // 绿色 - 第二名
    '#e6a23c', // 黄色 - 第三名
    '#f56c6c', // 红色 - 第四名
    '#909399'  // 灰色 - 第五名
  ]
  return colors[index] || '#909399'
}

// 组件挂载时加载示例
onMounted(() => {
  loadExample()
})
</script>

<style scoped>
.task2-container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 40px 20px;
}

.header-section {
  text-align: center;
  margin-bottom: 40px;
}

.header-section h2 {
  font-size: 36px;
  font-weight: 700;
  color: #065f46;
  margin-bottom: 12px;
  position: relative;
}

.header-section h2::after {
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

.description {
  color: #10b981;
  font-size: 18px;
  line-height: 1.6;
  max-width: 600px;
  margin: 0 auto;
}

.main-content {
  display: grid;
  gap: 40px;
}

.input-section, .result-section {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 32px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);
  transition: all 0.3s ease;
}

.input-section:hover, .result-section:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15);
}

.input-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 20px;
}

.input-header h3 {
  color: #065f46;
  font-size: 20px;
  font-weight: 600;
}

.button-group {
  margin-top: 20px;
  display: flex;
  gap: 16px;
}

.result-header {
  margin-bottom: 24px;
}

.result-header h3 {
  color: #065f46;
  font-size: 20px;
  font-weight: 600;
}

.main-prediction {
  margin-bottom: 32px;
}

.prediction-card {
  background: linear-gradient(135deg, #10b981 0%, #065f46 100%);
  color: white;
  padding: 32px;
  border-radius: 16px;
  text-align: center;
  box-shadow: 0 8px 32px rgba(16, 185, 129, 0.3);
  position: relative;
  overflow: hidden;
}

.prediction-card::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: radial-gradient(circle at 30% 70%, rgba(245, 158, 11, 0.2), transparent 50%);
  pointer-events: none;
}

.prediction-label {
  font-size: 16px;
  opacity: 0.9;
  margin-bottom: 12px;
  font-weight: 500;
}

.prediction-value {
  font-size: 32px;
  font-weight: 800;
  position: relative;
  z-index: 1;
}

.probability-section {
  margin-bottom: 32px;
}

.probability-section h4 {
  margin-bottom: 20px;
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
}

.probability-list {
  display: grid;
  gap: 16px;
}

.probability-item {
  background: rgba(255, 255, 255, 0.8);
  padding: 20px;
  border-radius: 12px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  transition: all 0.3s ease;
}

.probability-item:hover {
  border-color: rgba(16, 185, 129, 0.4);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
}

.probability-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.category-name {
  font-weight: 600;
  color: #065f46;
  font-size: 16px;
}

.probability-value {
  font-weight: 700;
  color: #10b981;
  font-size: 16px;
}

.text-review {
  background: rgba(255, 255, 255, 0.8);
  padding: 20px;
  border-radius: 12px;
  border-left: 4px solid #10b981;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.text-review h4 {
  margin-bottom: 16px;
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
}

.review-text {
  background: rgba(16, 185, 129, 0.05);
  padding: 16px;
  border-radius: 8px;
  border: 1px solid rgba(16, 185, 129, 0.1);
  color: #065f46;
  line-height: 1.6;
  font-size: 16px;
}

.error-section {
  margin-top: 20px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 12px;
  padding: 16px;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .input-header {
    flex-direction: column;
    gap: 12px;
  }

  .prediction-card {
    padding: 20px;
  }

  .prediction-value {
    font-size: 20px;
  }

  .probability-info {
    flex-direction: column;
    align-items: flex-start;
    gap: 4px;
  }
}
</style>
