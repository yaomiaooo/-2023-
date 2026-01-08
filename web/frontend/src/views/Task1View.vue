<template>
  <div class="task1-container">
    <div class="header-section">
      <h2>🩺 医学实体抽取</h2>
      <p class="description">
        从医学文本中自动识别和提取实体，包括疾病、症状、药物、医疗设备等9大类实体。
      </p>
    </div>

    <div class="main-content">
      <!-- 输入区域 -->
      <div class="input-section">
        <div class="input-header">
          <h3>输入医学文本</h3>
          <el-button type="primary" @click="loadExample" size="small">
            加载示例
          </el-button>
        </div>

        <el-input
          v-model="inputText"
          type="textarea"
          :rows="6"
          placeholder="请输入医学文本..."
          :disabled="loading"
        />

        <div class="button-group">
          <el-button
            type="primary"
            :loading="loading"
            @click="predictEntities"
            :disabled="!inputText.trim()"
          >
            开始识别
          </el-button>
            <el-button @click="clearAll">清空</el-button>
            <el-button
              v-if="results"
              type="info"
              @click="showDebugInfo = !showDebugInfo"
            >
              {{ showDebugInfo ? '隐藏' : '调试' }}
            </el-button>
        </div>
      </div>

      <!-- 结果区域 -->
      <div v-if="results" class="result-section">
        <div class="result-header">
          <h3>识别结果</h3>
          <el-tag type="success">共识别 {{ results.entity_count }} 个实体</el-tag>
        </div>

        <!-- 实体类型统计 -->
        <div class="entity-stats">
          <div class="stat-item" v-for="(count, type) in entityTypeStats" :key="type">
            <span class="type-name">{{ type }}</span>
            <el-tag size="small">{{ count }}</el-tag>
          </div>
        </div>

        <!-- 文本高亮显示 -->
        <div class="highlighted-text">
          <div class="text-content" v-html="highlightedText"></div>
        </div>

        <!-- 实体列表 -->
        <div class="entity-list">
          <h4>实体详情</h4>
          <div class="entity-item" v-for="(entity, index) in results.entities" :key="index">
            <div class="entity-info">
              <span class="entity-text">"{{ entity.text }}"</span>
              <el-tag :type="getTagType(entity.type)" size="small">{{ entity.type }}</el-tag>
            </div>
            <div class="entity-position">
              位置: {{ entity.start }}-{{ entity.end }}
            </div>
          </div>
        </div>

        <!-- 调试信息 -->
        <div v-if="showDebugInfo && results" class="debug-section">
          <h4>调试信息</h4>
          <div class="debug-content">
            <div class="debug-item">
              <strong>原始文本:</strong>
              <pre>{{ results.text }}</pre>
            </div>
            <div class="debug-item">
              <strong>实体数据 ({{ results.entities.length }} 个):</strong>
              <pre>{{ JSON.stringify(results.entities, null, 2) }}</pre>
            </div>
            <div class="debug-item">
              <strong>文本长度:</strong>
              <pre>原始文本: {{ results.text.length }} 字符</pre>
            </div>
            <div class="debug-item">
              <strong>高亮文本HTML:</strong>
              <pre>{{ highlightedText }}</pre>
            </div>
          </div>
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
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

// 响应式数据
const inputText = ref('')
const loading = ref(false)
const results = ref(null)
const error = ref('')
const examples = ref([])
const showDebugInfo = ref(false)

// 加载示例数据
const loadExample = async () => {
  try {
    const response = await axios.get('/api/task1/examples')
    examples.value = response.data.examples
    if (examples.value.length > 0) {
      inputText.value = examples.value[Math.floor(Math.random() * examples.value.length)]
    }
  } catch (err) {
    ElMessage.error('加载示例失败')
  }
}

// 执行实体识别
const predictEntities = async () => {
  if (!inputText.value.trim()) {
    ElMessage.warning('请输入文本')
    return
  }

  loading.value = true
  error.value = ''
  results.value = null

  try {
    const response = await axios.post('/api/task1/predict', {
      text: inputText.value
    })

    results.value = response.data
    ElMessage({
      message: `成功识别 ${response.data.entity_count} 个实体`,
      type: 'success',
      duration: 1500
    })
  } catch (err) {
    error.value = err.response?.data?.error || '识别失败，请重试'
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

// 实体类型统计
const entityTypeStats = computed(() => {
  if (!results.value) return {}

  const stats = {}
  results.value.entities.forEach(entity => {
    stats[entity.type] = (stats[entity.type] || 0) + 1
  })
  return stats
})

// 生成高亮文本
const highlightedText = computed(() => {
  if (!results.value) return inputText.value

  try {
    const text = results.value.text || inputText.value
    const entities = [...results.value.entities]

    // 按起始位置升序排序（从前到后处理）
    entities.sort((a, b) => a.start - b.start)

    // 过滤掉重叠的实体（保留第一个找到的）
    const filteredEntities = []
    let lastEnd = -1

    for (const entity of entities) {
      if (entity.start >= lastEnd && entity.end <= text.length && entity.start < entity.end) {
        filteredEntities.push(entity)
        lastEnd = entity.end
      }
    }

    let result = ''
    let lastIndex = 0

    filteredEntities.forEach(entity => {
      // 添加实体前的普通文本
      result += text.substring(lastIndex, entity.start)

      // 添加高亮实体
      const entityText = text.substring(entity.start, entity.end)
      const color = getEntityColor(entity.type)
      const highlightedEntity = `<span class="entity-highlight" style="background-color: ${color}; padding: 2px 4px; border-radius: 3px; margin: 0 1px;" title="${entity.type}">${entityText}</span>`

      result += highlightedEntity
      lastIndex = entity.end
    })

    // 添加剩余的文本
    result += text.substring(lastIndex)

    // 验证生成的HTML是否包含未闭合的标签
    const openTags = (result.match(/<span[^>]*>/g) || []).length
    const closeTags = (result.match(/<\/span>/g) || []).length

    if (openTags !== closeTags) {
      console.warn(`HTML标签不匹配: ${openTags} 个开始标签, ${closeTags} 个结束标签`)
      // 如果标签不匹配，返回纯文本
      return text
    }

    return result
  } catch (error) {
    console.error('高亮文本生成错误:', error)
    return results.value?.text || inputText.value
  }
})

// 获取实体颜色
const getEntityColor = (type) => {
  const colorMap = {
    '疾病(dis)': '#ffccc7',
    '症状(sym)': '#ffe7ba',
    '医疗程序(pro)': '#d9f7be',
    '医疗设备(equ)': '#bae7ff',
    '药物(dru)': '#efdbff',
    '医学检验项目(ite)': '#ffd6e7',
    '身体(bod)': '#fff1b8',
    '科室(dep)': '#b5f5ec',
    '微生物类(mic)': '#d6e4ff'
  }
  return colorMap[type] || '#f0f0f0'
}

// 获取标签类型
const getTagType = (type) => {
  const typeMap = {
    '疾病(dis)': 'danger',
    '症状(sym)': 'warning',
    '医疗程序(pro)': 'success',
    '医疗设备(equ)': 'info',
    '药物(dru)': 'primary'
  }
  return typeMap[type] || ''
}

// 组件挂载时加载示例
onMounted(() => {
  loadExample()
})
</script>

<style scoped>
.task1-container {
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
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 24px;
}

.result-header h3 {
  color: #065f46;
  font-size: 20px;
  font-weight: 600;
}

.entity-stats {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  margin-bottom: 24px;
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: rgba(16, 185, 129, 0.1);
  border-radius: 20px;
}

.type-name {
  font-weight: 500;
  color: #065f46;
  font-size: 14px;
}

.highlighted-text {
  background: rgba(255, 255, 255, 0.8);
  border-radius: 12px;
  padding: 20px;
  margin-bottom: 24px;
  line-height: 1.8;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.text-content {
  white-space: pre-wrap;
  word-wrap: break-word;
  color: #065f46;
  font-size: 16px;
}

.entity-highlight {
  border-radius: 4px;
  font-weight: 500;
}

.entity-list {
  margin-top: 24px;
}

.entity-list h4 {
  margin-bottom: 16px;
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
}

.entity-item {
  padding: 16px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  border-radius: 12px;
  margin-bottom: 12px;
  background: rgba(255, 255, 255, 0.6);
  transition: all 0.3s ease;
}

.entity-item:hover {
  border-color: rgba(16, 185, 129, 0.4);
  box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);
}

.entity-info {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.entity-text {
  font-weight: 500;
  color: #065f46;
  font-size: 16px;
  padding: 4px 8px;
  background: rgba(16, 185, 129, 0.1);
  border-radius: 6px;
}

.entity-position {
  font-size: 14px;
  color: #10b981;
  font-weight: 500;
}

.error-section {
  margin-top: 20px;
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.2);
  border-radius: 12px;
  padding: 16px;
}

.debug-section {
  margin-top: 20px;
  background: rgba(245, 158, 11, 0.1);
  border: 1px solid rgba(245, 158, 11, 0.2);
  border-radius: 12px;
  padding: 16px;
}

.debug-section h4 {
  color: #f59e0b;
  margin-bottom: 12px;
  font-size: 16px;
  font-weight: 600;
}

.debug-content {
  display: grid;
  gap: 12px;
}

.debug-item {
  background: white;
  border-radius: 8px;
  padding: 12px;
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.debug-item strong {
  color: #065f46;
  display: block;
  margin-bottom: 8px;
}

.debug-item pre {
  background: #f8f9fa;
  padding: 8px;
  border-radius: 4px;
  font-size: 12px;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-height: 200px;
  overflow-y: auto;
}

/* 响应式设计 */
@media (max-width: 768px) {
  .input-header, .result-header {
    flex-direction: column;
    gap: 12px;
  }

  .entity-stats {
    justify-content: center;
  }

  .entity-info {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
}
</style>
