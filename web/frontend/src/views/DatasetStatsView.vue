<template>
  <div class="dataset-stats-container">
    <!-- 页面头部 -->
    <div class="header-section">
      <h2>数据集统计</h2>
      <p>查看训练数据集的详细统计信息和分布情况</p>
    </div>

    <!-- 数据集切换 -->
    <div class="dataset-tabs">
      <el-radio-group v-model="activeDataset" @change="handleDatasetChange" :disabled="loading">
        <el-radio-button label="cmeee">CMeEE - 医学实体识别</el-radio-button>
        <el-radio-button label="chip_ctc">CHIP-CTC - 临床试验分类</el-radio-button>
      </el-radio-group>
      <div v-if="loading" class="switch-loading">
        <el-icon class="is-loading">
          <loading />
        </el-icon>
        <span>切换中...</span>
      </div>
    </div>

      <!-- 数据集基本信息
      <div class="dataset-info">
        <h3>{{ stats.dataset_name }}</h3>
        <p>{{ stats.description }}</p>
      </div> -->

    <!-- 数据特征分析 -->
    <div class="feature-analysis">
      <h3>数据特征分析</h3>

      <!-- 数据结构解析 -->
      <div class="analysis-section">
        <h4>📋 数据结构解析</h4>
        <div class="structure-grid">
          <div class="structure-card" v-if="activeDataset === 'cmeee'">
            <h5>实体识别任务 (CMeEE)</h5>
            <div class="structure-content">
              <div class="structure-format">
                <strong>样本格式：</strong>List[Dict]
              </div>
              <div class="structure-fields">
                <div class="field-item">
                  <span class="field-name">"text"</span>
                  <span class="field-desc">原始文本</span>
                </div>
                <div class="field-item">
                  <span class="field-name">"entities"</span>
                  <span class="field-desc">实体列表</span>
                </div>
                <div class="field-details">
                  <strong>实体属性：</strong>
                  <div class="sub-fields">
                    <span>• "start_idx" - 起始位置</span>
                    <span>• "end_idx" - 结束位置</span>
                    <span>• "type" - 实体类型</span>
                    <span>• "entity" - 实体文本</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div class="structure-card" v-if="activeDataset === 'chip_ctc'">
            <h5>文本分类任务 (CHIP-CTC)</h5>
            <div class="structure-content">
              <div class="structure-format">
                <strong>样本格式：</strong>Dict
              </div>
              <div class="structure-fields">
                <div class="field-item">
                  <span class="field-name">"text"</span>
                  <span class="field-desc">描述文本</span>
                </div>
                <div class="field-item">
                  <span class="field-name">"label"</span>
                  <span class="field-desc">类别标签 (44个预定义医学实体类别)</span>
                </div>
                <div class="field-item">
                  <span class="field-name">"id"</span>
                  <span class="field-desc">样本ID</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 长度分布分析 -->
      <div class="analysis-section">
        <h4>📏 长度分布分析</h4>
        <div class="length-analysis">
          <div class="length-card">
            <h5>文本长度分布</h5>
            <div class="length-stats">
              <div class="stat-item" v-if="activeDataset === 'cmeee'">
                <span class="stat-label">实体识别 (CMeEE)</span>
                <div class="stat-details">
                  <span>• 最短: 4字符</span>
                  <span>• 最长: 4870字符</span>
                  <span>• 平均: 54.15字符</span>
                  <span>• 分布: 多数集中在20-100字符，呈右偏态</span>
                </div>
              </div>
              <div class="stat-item" v-if="activeDataset === 'chip_ctc'">
                <span class="stat-label">文本分类 (CHIP-CTC)</span>
                <div class="stat-details">
                  <span>• 最短: 3字符</span>
                  <span>• 最长: 342字符</span>
                  <span>• 平均: 27.15字符</span>
                  <span>• 分布: 多数集中在10-50字符，呈右偏态</span>
                </div>
              </div>
            </div>
          </div>

          <div class="length-card" v-if="activeDataset === 'cmeee'">
            <h5>实体长度分布</h5>
            <div class="length-stats">
              <div class="stat-item">
                <span class="stat-label">实体长度统计</span>
                <div class="stat-details">
                  <span>• 最短: 1字符</span>
                  <span>• 最长: 139字符</span>
                  <span>• 平均: 5.09字符</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <!-- 类别/实体类型分布 -->
      <div class="analysis-section">
        <h4>🏷️ 类别/实体类型分布</h4>
        <div class="distribution-analysis">
          <div class="distribution-card" v-if="activeDataset === 'cmeee'">
            <h5>实体识别 (9类实体)</h5>
            <div class="distribution-alert">
              <el-alert
                title="分布极不平衡"
                description="实体类型分布严重不均衡，'bod（身体）'占比最高（24.84%），'dep（科室）'占比最低（0.36%）"
                type="warning"
                :closable="false"
                show-icon
              />
            </div>
            <div class="entity-type-highlights">
              <div class="highlight-item">
                <span class="highlight-label">最高频实体:</span>
                <el-tag type="success">bod（身体）- 24.84%</el-tag>
              </div>
              <div class="highlight-item">
                <span class="highlight-label">最低频实体:</span>
                <el-tag type="danger">dep（科室）- 0.36%</el-tag>
              </div>
            </div>
          </div>

          <div class="distribution-card" v-if="activeDataset === 'chip_ctc'">
            <h5>文本分类 (44类标签)</h5>
            <div class="distribution-alert">
              <el-alert
                title="分布严重不平衡"
                description="类别分布极不均衡，高频类别样本量远高于低频类别，存在模型偏向高频类别的风险"
                type="warning"
                :closable="false"
                show-icon
              />
            </div>
            <div v-if="stats.category_distribution && stats.category_distribution.overall" class="category-highlights">
              <div class="highlight-item">
                <span class="highlight-label">类别数量:</span>
                <el-tag type="info">{{ stats.category_distribution && stats.category_distribution.overall ? Object.keys(stats.category_distribution.overall).length : 0 }}个预定义医学实体类别</el-tag>
              </div>
              <div class="highlight-item">
                <span class="highlight-label">分布特征:</span>
                <el-tag type="warning">高频类别样本量显著高于低频类别</el-tag>
              </div>

            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 加载状态 -->
    <div v-if="loading" class="loading-section">
      <el-skeleton animated>
        <template #template>
          <div class="stats-grid">
            <el-skeleton-item variant="rect" style="width: 100%; height: 200px;" />
            <el-skeleton-item variant="rect" style="width: 100%; height: 200px;" />
            <el-skeleton-item variant="rect" style="width: 100%; height: 200px;" />
          </div>
        </template>
      </el-skeleton>
    </div>

    <!-- 统计内容 -->
    <div v-else-if="stats" class="stats-content">


      <!-- 基本统计卡片 -->
      <div class="stats-grid">
        <div class="stat-card">
          <div class="stat-icon">
            📊
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.basic_stats.total_samples }}</div>
            <div class="stat-label">总样本数</div>
          </div>
        </div>

        <div class="stat-card" v-if="activeDataset === 'cmeee'">
          <div class="stat-icon">
            🎯
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.basic_stats.total_entities }}</div>
            <div class="stat-label">总实体数</div>
          </div>
        </div>

        <div class="stat-card" v-if="activeDataset === 'chip_ctc'">
          <div class="stat-icon">
            🏷️
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.basic_stats.num_categories }}</div>
            <div class="stat-label">类别数量</div>
          </div>
        </div>

        <div class="stat-card">
          <div class="stat-icon">
            📏
          </div>
          <div class="stat-content">
            <div class="stat-value">{{ stats.basic_stats.avg_text_length }}</div>
            <div class="stat-label">平均文本长度</div>
          </div>
        </div>
      </div>

      <!-- 详细统计 -->
      <div class="detailed-stats">
        <!-- CMeEE数据集统计 -->
        <div v-if="activeDataset === 'cmeee'" class="stats-section">


          <h4>实体类型统计</h4>
          <div class="entity-types-grid">
            <div
              v-for="(count, type) in stats.entity_type_distribution"
              :key="type"
              class="entity-type-item"
            >
              <span class="entity-type">{{ type }}</span>
              <el-tag size="small">{{ count }}</el-tag>
            </div>
          </div>

          <h4>文本长度分布</h4>
          <div class="length-distribution">
            <div
              v-for="(count, index) in stats.text_length_distribution.counts"
              :key="index"
              class="length-bar"
            >
              <div class="length-label">
                {{ getLengthLabel(index) }}
              </div>
              <div class="progress-container">
                <el-progress
                  :percentage="getPercentage(count, stats.basic_stats.total_samples)"
                  :show-text="false"
                  :stroke-width="8"
                  color="#10b981"
                />
              </div>
              <div class="length-count">{{ count }}</div>
            </div>
          </div>
        </div>

        <!-- CHIP-CTC数据集统计 -->
        <div v-else class="stats-section">
          <h4>数据分割详情</h4>
          <div class="split-info">
            <div class="split-item">
              <span class="split-label">训练集</span>
              <el-tag type="success">{{ stats.basic_stats.train_samples }}</el-tag>
            </div>
            <div class="split-item">
              <span class="split-label">测试集</span>
              <el-tag type="warning">{{ stats.basic_stats.test_samples }}</el-tag>
            </div>
          </div>

          <h4>CHIP-CTC类别详情</h4>
          <div class="category-search">
            <el-input
              v-model="categorySearch"
              placeholder="搜索类别、中文标签或英文标签..."
              clearable
              size="small"
            >
              <template #prefix>
                <el-icon><Search /></el-icon>
              </template>
            </el-input>
          </div>
          <div class="category-details">
            <div v-if="categorySearch.trim()" class="search-result-info">
              <el-text size="small" type="info">
                找到 {{ filteredCategories.length }} 个相关类别
              </el-text>
            </div>
            <div class="category-table-container">
              <table class="category-table">
                <thead>
                  <tr>
                    <th>主题组</th>
                    <th>中文标签名</th>
                    <th>英文标签名</th>
                    <th>示例</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="category in filteredCategories" :key="category.label">
                    <td class="topic-group">{{ category.topicGroup }}</td>
                    <td class="chinese-label">{{ category.chineseLabel }}</td>
                    <td class="english-label">{{ category.label }}</td>
                    <td class="examples">
                      <div class="example-list">
                        <div v-for="(example, index) in category.examples" :key="index" class="example-item">
                          {{ example }}
                        </div>
                      </div>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>



          <h4>文本长度分布</h4>
          <div class="length-distribution">
            <div
              v-for="(count, index) in stats.text_length_distribution.counts"
              :key="index"
              class="length-bar"
            >
              <div class="length-label">
                {{ getChipCtcLengthLabel(index) }}
              </div>
              <div class="progress-container">
                <el-progress
                  :percentage="getPercentage(count, stats.basic_stats.total_samples)"
                  :show-text="false"
                  :stroke-width="8"
                  color="#14b8a6"
                />
              </div>
              <div class="length-count">{{ count }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 错误状态 -->
    <div v-else-if="error" class="error-section">
      <el-alert
        :title="error"
        type="error"
        :closable="false"
        show-icon
      />
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, computed } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { Loading, Search } from '@element-plus/icons-vue'

// 响应式数据
const activeDataset = ref('cmeee')
const stats = ref(null)
const loading = ref(false)
const error = ref('')
const categorySearch = ref('')

// 切换数据集
const switchDataset = async () => {
  await loadStats()
}

// 处理数据集切换
const handleDatasetChange = async (value) => {
  try {
    activeDataset.value = value
    await switchDataset()
  } catch (err) {
    console.error('Dataset switch error:', err)
    ElMessage.error('切换数据集失败，请重试')
  }
}

// 加载统计数据
const loadStats = async () => {
  loading.value = true
  error.value = ''

  try {
    const endpoint = activeDataset.value === 'cmeee'
      ? '/api/dataset/stats/cmeee'
      : '/api/dataset/stats/chip_ctc'

    const response = await axios.get(endpoint)
    stats.value = response.data.stats
  } catch (err) {
    error.value = err.response?.data?.error || '加载统计数据失败'
    ElMessage.error(error.value)
    // 发生错误时不清空stats，保持上一次成功的数据
  } finally {
    loading.value = false
  }
}

// 获取长度标签
const getLengthLabel = (index) => {
  const bins = ['≤50', '50-100', '100-150', '150-200', '200-300', '300-500', '500-1000', '>1000']
  return bins[index] || ''
}

// 获取CHIP-CTC长度标签
const getChipCtcLengthLabel = (index) => {
  const bins = ['≤10', '10-20', '20-30', '30-50', '50-100', '100-200', '200-500', '>500']
  return bins[index] || ''
}

// 计算百分比
const getPercentage = (count, total) => {
  return Math.round((count / total) * 100)
}

// 过滤类别
const filteredCategories = computed(() => {
  if (!categorySearch.value.trim()) {
    return chipCtcCategories
  }

  const search = categorySearch.value.toLowerCase()
  return chipCtcCategories.filter(category =>
    category.topicGroup.toLowerCase().includes(search) ||
    category.chineseLabel.toLowerCase().includes(search) ||
    category.label.toLowerCase().includes(search) ||
    category.examples.some(example => example.toLowerCase().includes(search))
  )
})

// 计算类别百分比
const getCategoryPercentage = (count, total) => {
  if (!total || total === 0) return 0
  return Math.round((count / total) * 100 * 100) / 100 // 保留两位小数
}

// 获取类别颜色
const getCategoryColor = (category) => {
  try {
    if (!stats.value || !stats.value.category_distribution || !stats.value.category_distribution.overall || !stats.value.basic_stats) {
      return '#6b7280' // 默认灰色
    }

    const count = stats.value.category_distribution.overall[category]
    const total = stats.value.basic_stats.total_samples

    if (!count || !total || total === 0) return '#6b7280'

    const percentage = (count / total) * 100

    if (percentage >= 5) return '#10b981' // 高频 - 绿色
    if (percentage >= 2) return '#14b8a6' // 中频 - 青色
    return '#6b7280' // 低频 - 灰色
  } catch (error) {
    console.warn('Error getting category color:', error)
    return '#6b7280'
  }
}

// 获取实体颜色类名
const getEntityColorClass = (entityType) => {
  try {
    if (!stats.value || !stats.value.entity_type_distribution || !stats.value.basic_stats) {
      return 'color-low'
    }

    const count = stats.value.entity_type_distribution[entityType]
    const total = stats.value.basic_stats.total_entities

    if (!count || !total || total === 0) return 'color-low'

    const percentage = (count / total) * 100

    if (percentage >= 15) return 'color-high' // 高频实体 - 绿色
    if (percentage >= 5) return 'color-medium' // 中频实体 - 青色
    return 'color-low' // 低频实体 - 灰色
  } catch (error) {
    console.warn('Error getting entity color class:', error)
    return 'color-low'
  }
}

// 计算实体百分比
const getEntityPercentage = (count, total) => {
  if (!total || total === 0) return 0
  return Math.round((count / total) * 100 * 100) / 100 // 保留两位小数
}

// CHIP-CTC类别详细信息
const chipCtcCategories = [
  {
    topicGroup: 'Health Status',
    chineseLabel: '疾病',
    label: 'Disease',
    examples: [
      '1.胰腺炎病史'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '症状(患者感受)',
    label: 'Symptom',
    examples: [
      '1.以颈痛为主诉者'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '体征(医生检测）',
    label: 'Sign',
    examples: [
      '1.顽固性大量腹水'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '怀孕相关',
    label: 'Pregnancy-related Activity',
    examples: [
      '1.孕妇和哺乳期妇女'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '肿瘤进展',
    label: 'Neoplasm Status',
    examples: [
      '1.存在局部淋巴结侵犯'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '疾病分期',
    label: 'Non-Neoplasm Disease Stage',
    examples: [
      '1.患者病情处于不稳定期'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '过敏耐受',
    label: 'Allergy Intolerance',
    examples: [
      '1.既往有药物过敏史者'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '器官组织状态',
    label: 'Organ or Tissue Status',
    examples: [
      '1.肾功能正常'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '预期寿命',
    label: 'Life Expectancy',
    examples: [
      '1.预期复发后30天内可能会出现早期死亡的患者'
    ]
  },
  {
    topicGroup: 'Health Status',
    chineseLabel: '口腔相关',
    label: 'Oral related',
    examples: [
      '1.口腔卫生状况较差'
    ]
  },
  {
    topicGroup: 'Treatment or Health Care',
    chineseLabel: '药物',
    label: 'Pharmaceutical Substance or Drug',
    examples: [
      '1.有精神或神经科药物服用史者'
    ]
  },
  {
    topicGroup: 'Treatment or Health Care',
    chineseLabel: '治疗或手术',
    label: 'Therapy or Surgery',
    examples: [
      '1.脊柱外科手术史'
    ]
  },
  {
    topicGroup: 'Treatment or Health Care',
    chineseLabel: '设备',
    label: 'Device',
    examples: [
      '1.用球囊与支架对吻技术'
    ]
  },
  {
    topicGroup: 'Treatment or Health Care',
    chineseLabel: '护理',
    label: 'Nursing',
    examples: [
      '1.卧床制动患者≥72小时'
    ]
  },
  {
    topicGroup: 'Diagnostic or Lab Test',
    chineseLabel: '诊断',
    label: 'Diagnostic',
    examples: [
      '1.符合肩颈、腰腿痛诊断标准'
    ]
  },
  {
    topicGroup: 'Diagnostic or Lab Test',
    chineseLabel: '实验室检查',
    label: 'Laboratory Examinations',
    examples: [
      '1.左室射血分数（LVEF）≥50%'
    ]
  },
  {
    topicGroup: 'Diagnostic or Lab Test',
    chineseLabel: '风险评估',
    label: 'Risk Assessment',
    examples: [
      '1.ASA分级Ⅰ～Ⅱ级'
    ]
  },
  {
    topicGroup: 'Diagnostic or Lab Test',
    chineseLabel: '受体状态',
    label: 'Receptor Status',
    examples: [
      '1.移植均为新鲜或冻存优质胚胎（6C II级以上）'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '年龄',
    label: 'Age',
    examples: [
      '1.年龄65~75 岁'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '特殊病人特征',
    label: 'Special Patient Characteristic',
    examples: [
      '1.夜磨牙、紧咬牙等不良习惯'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '读写能力',
    label: 'Literacy',
    examples: [
      '1.能够熟练阅读，使用中文'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '性别',
    label: 'Gender',
    examples: [
      '1.性别不限'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '教育情况',
    label: 'Education',
    examples: [
      '1.小学以上文化程度'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '居住情况',
    label: 'Address',
    examples: [
      '1.地域：中国北方地区'
    ]
  },
  {
    topicGroup: 'Demographic Characteristics',
    chineseLabel: '种族',
    label: 'Ethnicity',
    examples: [
      '1.中国籍患者'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '知情同意',
    label: 'Consent',
    examples: [
      '1.签署知情同意书'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '参与其它试验',
    label: 'Enrollment in other studies',
    examples: [
      '1.正在参加影响本研究结果评价的其它临床试验者'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '研究者决定',
    label: 'Researcher Decision',
    examples: [
      '1.研究者判断不适合参加本研究的其他情况'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '能力',
    label: 'Capacity',
    examples: [
      '1.不能平卧或半卧位的患者'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '伦理审查',
    label: 'Ethical Audit',
    examples: [
      '1.伦理审核未通过者'
    ]
  },
  {
    topicGroup: 'Ethical Consideration',
    chineseLabel: '依存性',
    label: 'Compliance with Protocol',
    examples: [
      '1.依从性差的患者'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '成瘾行为',
    label: 'Addictive Behavior',
    examples: [
      '1.有药物成瘾的证据'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '睡眠',
    label: 'Bedtime',
    examples: [
      '1.昼夜颠倒的生活方式，或不规律的睡眠模式者'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '锻炼',
    label: 'Exercise',
    examples: [
      '1.平时无运动锻炼习惯'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '饮食',
    label: 'Diet',
    examples: [
      '1.没有咖啡饮用习惯'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '酒精使用',
    label: 'Alcohol Consumer',
    examples: [
      '1.每周饮酒超过28个单位酒精（1单位=285ml啤酒或25ml烈酒或125ml葡萄酒）'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '性取向',
    label: 'Sexual related',
    examples: [
      '1.阳性方为男男同性性行为者'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '吸烟状况',
    label: 'Smoking Status',
    examples: [
      '1.吸烟史：大于20包年'
    ]
  },
  {
    topicGroup: 'Lifestyle Choice',
    chineseLabel: '献血',
    label: 'Blood Donation',
    examples: [
      '1.试验前3个月内参与献血者'
    ]
  },
  {
    topicGroup: 'Data or Patient Source',
    chineseLabel: '病例来源',
    label: 'Encounter',
    examples: [
      '1.在我院神经康复科住院的患者'
    ]
  },
  {
    topicGroup: 'Data or Patient Source',
    chineseLabel: '残疾群体',
    label: 'Disabilities',
    examples: [
      '1.单纯视力残疾人'
    ]
  },
  {
    topicGroup: 'Data or Patient Source',
    chineseLabel: '健康群体',
    label: 'Healthy',
    examples: [
      '1.身体，精神发育正常'
    ]
  },
  {
    topicGroup: 'Data or Patient Source',
    chineseLabel: '数据可及性',
    label: 'Data Accessible',
    examples: [
      '1.相关临床资料完整'
    ]
  },
  {
    topicGroup: 'Other',
    chineseLabel: '含有多类别的语句',
    label: 'Multiple',
    examples: [
      '1.严重精神疾病；有酗酒、药瘾或者其他不适合参加研究者'
    ]
  }
]

// 组件挂载时加载数据
onMounted(() => {
  loadStats()
})

// 组件卸载时清理状态
onUnmounted(() => {
  // 清理状态
})
</script>

<style scoped>
.dataset-stats-container {
  max-width: 1200px;
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

.header-section p {
  color: #10b981;
  font-size: 18px;
  line-height: 1.6;
  max-width: 600px;
  margin: 0 auto;
}

.dataset-tabs {
  display: flex;
  justify-content: center;
  align-items: center;
  gap: 20px;
  margin-bottom: 40px;
}

.switch-loading {
  display: flex;
  align-items: center;
  gap: 8px;
  color: #10b981;
  font-size: 14px;
}

.switch-loading .el-icon {
  font-size: 16px;
}

.loading-section {
  margin-top: 40px;
}

.stats-content {
  animation: fadeIn 0.5s ease-in-out;
}

.dataset-info {
  text-align: center;
  margin-bottom: 40px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.dataset-info h3 {
  color: #065f46;
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 8px;
}

.dataset-info p {
  color: #10b981;
  font-size: 16px;
}

.feature-analysis {
  margin-top: 40px;
  margin-bottom: 40px;
  padding: 32px;
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  backdrop-filter: blur(10px);
}

.feature-analysis h3 {
  color: #065f46;
  font-size: 24px;
  font-weight: 600;
  margin-bottom: 32px;
  text-align: center;
  position: relative;
}

.feature-analysis h3::after {
  content: '';
  position: absolute;
  bottom: -8px;
  left: 50%;
  transform: translateX(-50%);
  width: 80px;
  height: 3px;
  background: linear-gradient(90deg, #10b981, #34d399);
  border-radius: 2px;
}

.analysis-section {
  margin-bottom: 40px;
}

.analysis-section h4 {
  color: #065f46;
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.structure-grid {
  display: grid;
  gap: 24px;
}

.structure-card {
  background: rgba(16, 185, 129, 0.05);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.structure-card h5 {
  color: #065f46;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.structure-content {
  display: grid;
  gap: 16px;
}

.structure-format {
  color: #10b981;
  font-size: 16px;
  font-weight: 500;
}

.structure-fields {
  display: grid;
  gap: 12px;
}

.field-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 8px 12px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 6px;
}

.field-name {
  color: #065f46;
  font-weight: 600;
  font-family: monospace;
  background: rgba(16, 185, 129, 0.1);
  padding: 2px 6px;
  border-radius: 4px;
}

.field-desc {
  color: #6b7280;
  font-size: 14px;
}

.field-details {
  color: #065f46;
  font-size: 14px;
  font-weight: 500;
}

.sub-fields {
  display: grid;
  gap: 4px;
  margin-top: 8px;
  padding-left: 16px;
}

.sub-fields span {
  color: #6b7280;
  font-size: 13px;
}

.length-analysis {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 24px;
}

.length-card {
  background: rgba(20, 184, 166, 0.05);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid rgba(20, 184, 166, 0.2);
}

.length-card h5 {
  color: #0f766e;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.length-stats {
  display: grid;
  gap: 16px;
}

.stat-item {
  padding: 16px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  border: 1px solid rgba(20, 184, 166, 0.1);
}

.stat-label {
  color: #0f766e;
  font-weight: 600;
  font-size: 16px;
  margin-bottom: 12px;
  display: block;
}

.stat-details {
  display: grid;
  gap: 6px;
}

.stat-details span {
  color: #6b7280;
  font-size: 14px;
  line-height: 1.4;
}

.distribution-analysis {
  display: grid;
  gap: 24px;
}

.distribution-card {
  background: rgba(245, 158, 11, 0.05);
  border-radius: 12px;
  padding: 24px;
  border: 1px solid rgba(245, 158, 11, 0.2);
}

.distribution-card h5 {
  color: #92400e;
  font-size: 18px;
  font-weight: 600;
  margin-bottom: 16px;
  display: flex;
  align-items: center;
  gap: 8px;
}

.distribution-alert {
  margin-bottom: 20px;
}

.entity-type-highlights,
.category-highlights {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 16px;
}

.highlight-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  border: 1px solid rgba(245, 158, 11, 0.1);
}

.highlight-label {
  color: #92400e;
  font-weight: 500;
  font-size: 14px;
  white-space: nowrap;
}

.stats-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
  gap: 20px;
  margin-bottom: 40px;
}

.stat-card {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 24px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  backdrop-filter: blur(10px);
  box-shadow: 0 8px 32px rgba(16, 185, 129, 0.1);
  transition: all 0.3s ease;
  display: flex;
  align-items: center;
  gap: 16px;
}

.stat-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 40px rgba(16, 185, 129, 0.15);
}

.stat-icon {
  font-size: 32px;
  opacity: 0.8;
}

.stat-content {
  flex: 1;
}

.stat-value {
  font-size: 32px;
  font-weight: 800;
  color: #065f46;
  margin-bottom: 4px;
}

.stat-label {
  font-size: 14px;
  color: #10b981;
  font-weight: 500;
}

.detailed-stats {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 16px;
  padding: 32px;
  border: 1px solid rgba(16, 185, 129, 0.2);
  backdrop-filter: blur(10px);
}

.stats-section h4 {
  color: #065f46;
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 20px;
  padding-bottom: 8px;
  border-bottom: 2px solid rgba(16, 185, 129, 0.3);
}

/* 实体分布图表 */
.entity-distribution {
  margin-bottom: 40px;
}

.distribution-chart {
  display: grid;
  gap: 12px;
  margin-bottom: 24px;
  max-height: 300px;
  overflow-y: auto;
  padding-right: 8px;
}

.distribution-chart::-webkit-scrollbar {
  width: 6px;
}

.distribution-chart::-webkit-scrollbar-track {
  background: rgba(16, 185, 129, 0.1);
  border-radius: 3px;
}

.distribution-chart::-webkit-scrollbar-thumb {
  background: rgba(16, 185, 129, 0.3);
  border-radius: 3px;
}

.distribution-chart::-webkit-scrollbar-thumb:hover {
  background: rgba(16, 185, 129, 0.5);
}

.entity-name {
  color: #065f46;
  font-weight: 500;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.entity-count {
  color: #10b981;
  font-weight: 600;
  font-size: 14px;
  margin-left: 8px;
  flex-shrink: 0;
}

.entity-types-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 16px;
  margin-bottom: 40px;
}

.entity-type-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 12px 16px;
  background: rgba(16, 185, 129, 0.05);
  border-radius: 8px;
  border: 1px solid rgba(16, 185, 129, 0.2);
}

.entity-type {
  color: #065f46;
  font-weight: 500;
}

.length-distribution {
  display: grid;
  gap: 16px;
}

.length-bar {
  display: grid;
  grid-template-columns: 120px 1fr 60px;
  align-items: center;
  gap: 16px;
}

.length-label {
  color: #065f46;
  font-weight: 500;
  font-size: 14px;
}

.progress-container {
  min-width: 0;
}

.length-count {
  color: #10b981;
  font-weight: 600;
  text-align: right;
}

.split-info {
  display: flex;
  gap: 32px;
  margin-bottom: 40px;
}

.split-item {
  display: flex;
  align-items: center;
  gap: 12px;
}

.split-label {
  color: #065f46;
  font-weight: 500;
}

.category-distribution {
  margin-bottom: 40px;
}

.distribution-chart {
  display: grid;
  gap: 12px;
  margin-bottom: 24px;
  max-height: 400px;
  overflow-y: auto;
  padding-right: 8px;
}

.distribution-chart::-webkit-scrollbar {
  width: 6px;
}

.distribution-chart::-webkit-scrollbar-track {
  background: rgba(20, 184, 166, 0.1);
  border-radius: 3px;
}

.distribution-chart::-webkit-scrollbar-thumb {
  background: rgba(20, 184, 166, 0.3);
  border-radius: 3px;
}

.distribution-chart::-webkit-scrollbar-thumb:hover {
  background: rgba(20, 184, 166, 0.5);
}

.chart-item {
  display: grid;
  grid-template-columns: 200px 1fr;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 8px;
  border: 1px solid rgba(20, 184, 166, 0.1);
  transition: all 0.3s ease;
}

.chart-item:hover {
  border-color: rgba(20, 184, 166, 0.3);
  box-shadow: 0 2px 8px rgba(20, 184, 166, 0.1);
}

.chart-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  min-width: 0;
}

.category-name {
  color: #0f766e;
  font-weight: 500;
  font-size: 14px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  flex: 1;
}

.category-count {
  color: #14b8a6;
  font-weight: 600;
  font-size: 14px;
  margin-left: 8px;
  flex-shrink: 0;
}

.chart-bar {
  min-width: 0;
}

.custom-progress {
  width: 100%;
  height: 12px;
  background-color: rgba(0, 0, 0, 0.1);
  border-radius: 6px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 6px;
  transition: width 0.3s ease;
}

.progress-fill.color-high {
  background-color: #10b981;
}

.progress-fill.color-medium {
  background-color: #14b8a6;
}

.progress-fill.color-low {
  background-color: #6b7280;
}

/* CHIP-CTC类别搜索 */
.category-search {
  margin-bottom: 20px;
  max-width: 400px;
}

.category-search .el-input {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 8px;
  border: 1px solid rgba(20, 184, 166, 0.2);
}

.category-search .el-input:hover {
  border-color: rgba(20, 184, 166, 0.4);
}

.search-result-info {
  margin-bottom: 12px;
  padding: 8px 12px;
  background: rgba(20, 184, 166, 0.1);
  border-radius: 6px;
  border: 1px solid rgba(20, 184, 166, 0.2);
}

/* CHIP-CTC类别详情表格 */
.category-details {
  margin-bottom: 40px;
}

.category-table-container {
  background: rgba(255, 255, 255, 0.9);
  border-radius: 12px;
  overflow: hidden;
  border: 1px solid rgba(20, 184, 166, 0.2);
  box-shadow: 0 4px 12px rgba(20, 184, 166, 0.1);
}

.category-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 14px;
}

.category-table thead {
  background: linear-gradient(135deg, #14b8a6 0%, #0f766e 100%);
  color: white;
}

.category-table th {
  padding: 16px 12px;
  text-align: left;
  font-weight: 600;
  border-bottom: 2px solid rgba(20, 184, 166, 0.3);
}

.category-table td {
  padding: 12px;
  border-bottom: 1px solid rgba(20, 184, 166, 0.1);
  vertical-align: top;
}

.category-table tbody tr:nth-child(even) {
  background: rgba(20, 184, 166, 0.02);
}

.category-table tbody tr:hover {
  background: rgba(20, 184, 166, 0.05);
  transition: background-color 0.2s ease;
}

.topic-group {
  font-weight: 600;
  color: #065f46;
  background: rgba(16, 185, 129, 0.08);
  border-left: 4px solid #10b981;
  padding-left: 8px;
  min-width: 140px;
}

.chinese-label {
  font-weight: 500;
  color: #0f766e;
  background: rgba(20, 184, 166, 0.08);
  min-width: 120px;
}

.english-label {
  font-family: 'Courier New', monospace;
  color: #92400e;
  background: rgba(245, 158, 11, 0.08);
  font-weight: 500;
  min-width: 120px;
}

.examples {
  background: rgba(255, 255, 255, 0.5);
  max-width: 400px;
}

.example-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.example-item {
  color: #6b7280;
  font-size: 13px;
  line-height: 1.4;
  padding: 4px 8px;
  background: rgba(255, 255, 255, 0.8);
  border-radius: 4px;
  border-left: 2px solid rgba(20, 184, 166, 0.3);
}

.example-item:hover {
  background: rgba(20, 184, 166, 0.1);
}

.chart-legend {
  display: flex;
  justify-content: center;
  gap: 24px;
  padding: 16px;
  background: rgba(20, 184, 166, 0.05);
  border-radius: 8px;
  border: 1px solid rgba(20, 184, 166, 0.1);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.legend-color {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}

.legend-text {
  color: #0f766e;
  font-size: 14px;
  font-weight: 500;
}

.error-section {
  margin-top: 40px;
}

@keyframes fadeIn {
  from {
    opacity: 0;
    transform: translateY(20px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

/* 响应式设计 */
@media (max-width: 768px) {
  .dataset-stats-container {
    padding: 20px 16px;
  }

  .header-section h2 {
    font-size: 28px;
  }

  .stats-grid {
    grid-template-columns: 1fr;
  }

  .stat-card {
    padding: 16px;
  }

  .stat-value {
    font-size: 24px;
  }

  .detailed-stats {
    padding: 20px;
  }

  .entity-types-grid {
    grid-template-columns: 1fr;
  }

  .distribution-chart {
    max-height: 250px;
  }

  .category-list {
    grid-template-columns: 1fr;
  }

  .split-info {
    flex-direction: column;
    gap: 16px;
  }

  .length-bar {
    grid-template-columns: 80px 1fr 40px;
    gap: 12px;
  }

  .feature-analysis {
    padding: 20px;
  }

  .structure-grid {
    grid-template-columns: 1fr;
  }

  .length-analysis {
    grid-template-columns: 1fr;
  }

  .entity-type-highlights,
  .category-highlights {
    grid-template-columns: 1fr;
  }

  .highlight-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }

  .field-item {
    flex-direction: column;
    align-items: flex-start;
    gap: 6px;
  }

  .sub-fields {
    padding-left: 12px;
  }

  .chart-item {
    grid-template-columns: 1fr;
    gap: 12px;
  }

  .chart-label {
    justify-content: flex-start;
    gap: 12px;
  }

  .chart-legend {
    flex-direction: column;
    gap: 12px;
    align-items: flex-start;
  }

  .distribution-chart {
    max-height: 300px;
  }

  .category-table-container {
    overflow-x: auto;
    -webkit-overflow-scrolling: touch;
  }

  .category-table {
    min-width: 800px;
    font-size: 12px;
  }

  .category-table th,
  .category-table td {
    padding: 8px 6px;
  }

  .topic-group,
  .chinese-label,
  .english-label {
    min-width: 100px;
  }

  .examples {
    max-width: 300px;
  }

  .example-item {
    font-size: 11px;
    padding: 3px 6px;
  }
}
</style>
