import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

class RotatoryPositionEncoder(nn.Module):
    """旋转位置编码实现"""
    def __init__(self, embedding_dim, max_position_length=512):
        super(RotatoryPositionEncoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.max_position_length = max_position_length

        # 计算频率倒数
        inverse_frequencies = 1.0 / (10000 ** (torch.arange(0, embedding_dim, 2).float() / embedding_dim))
        self.register_buffer('inverse_frequencies', inverse_frequencies)

        # 创建位置序列
        position_indices = torch.arange(0, max_position_length, dtype=torch.float)
        self.register_buffer('position_indices', position_indices)
        
    def forward(self, input_tensor):
        # 支持3维 [batch, seq_len, hidden] 或4维 [batch, heads, seq_len, hidden] 输入
        if input_tensor.dim() == 4:
            # 处理多头注意力情况
            batch_count, head_count, sequence_length, hidden_dimension = input_tensor.shape

            # 生成位置编码矩阵
            position_encoding = torch.outer(self.position_indices[:sequence_length], self.inverse_frequencies)
            sine_encoding = torch.sin(position_encoding)
            cosine_encoding = torch.cos(position_encoding)

            # 扩展编码维度
            sine_encoding = sine_encoding.repeat_interleave(2, dim=-1)
            cosine_encoding = cosine_encoding.repeat_interleave(2, dim=-1)

            # 扩展到批次和头数维度
            sine_encoding = sine_encoding.unsqueeze(0).expand(batch_count, head_count, -1, -1)
            cosine_encoding = cosine_encoding.unsqueeze(0).expand(batch_count, head_count, -1, -1)

            # 执行旋转操作
            even_components = input_tensor[..., ::2]
            odd_components = input_tensor[..., 1::2]
            rotated_components = torch.stack([-odd_components, even_components], dim=-1).reshape(batch_count, head_count, sequence_length, hidden_dimension)

            return input_tensor * cosine_encoding + rotated_components * sine_encoding
        else:
            # 处理标准序列输入
            batch_count, sequence_length, hidden_dimension = input_tensor.shape

            # 生成位置编码
            position_encoding = torch.outer(self.position_indices[:sequence_length], self.inverse_frequencies)
            sine_encoding = torch.sin(position_encoding)
            cosine_encoding = torch.cos(position_encoding)

            # 扩展编码矩阵
            sine_encoding = sine_encoding.repeat_interleave(2, dim=-1)
            cosine_encoding = cosine_encoding.repeat_interleave(2, dim=-1)

            # 应用旋转变换
            even_components = input_tensor[..., ::2]
            odd_components = input_tensor[..., 1::2]
            rotated_components = torch.stack([-odd_components, even_components], dim=-1).reshape(batch_count, sequence_length, hidden_dimension)

            return input_tensor * cosine_encoding + rotated_components * sine_encoding

class OptimizedGlobalPointer(nn.Module):
    """优化的全局指针网络层"""
    def __init__(self, hidden_dimension, head_count, head_dimension=64, enable_rope=True):
        super(OptimizedGlobalPointer, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.head_count = head_count
        self.head_dimension = head_dimension
        self.enable_rope = enable_rope

        if self.enable_rope:
            self.position_encoder = RotatoryPositionEncoder(embedding_dim=head_dimension, max_position_length=512)

        # 线性变换层，将隐藏状态映射到查询和键向量
        self.projection_layer = nn.Linear(hidden_dimension, head_count * head_dimension * 2)
        
    def forward(self, hidden_states, attention_weights=None):
        # 输入: [batch_size, seq_len, hidden_size]
        batch_size, sequence_length, hidden_dimension = hidden_states.shape

        # 通过线性层生成查询和键向量
        projected_states = self.projection_layer(hidden_states)
        projected_states = projected_states.reshape(batch_size, sequence_length, self.head_count, 2, self.head_dimension)
        query_vectors, key_vectors = projected_states[..., 0, :], projected_states[..., 1, :]  # [batch, seq_len, heads, head_dim]

        # 应用旋转位置编码
        if self.enable_rope:
            query_vectors = self.position_encoder(query_vectors.transpose(1, 2)).transpose(1, 2)
            key_vectors = self.position_encoder(key_vectors.transpose(1, 2)).transpose(1, 2)

        # 计算注意力权重矩阵
        query_vectors = query_vectors / (self.head_dimension ** 0.5)
        attention_scores = torch.einsum('bnhd,bmhd->bhnm', query_vectors, key_vectors)  # [batch, heads, seq_len, seq_len]

        # 应用注意力掩码
        if attention_weights is not None:
            expanded_mask = attention_weights.unsqueeze(1).unsqueeze(2).repeat(1, self.head_count, sequence_length, 1)
            attention_scores = attention_scores.masked_fill(expanded_mask == 0, -1e12)

        # 对角线屏蔽，防止自指预测
        identity_mask = torch.eye(sequence_length, device=attention_scores.device).unsqueeze(0).unsqueeze(1)
        attention_scores = attention_scores.masked_fill(identity_mask == 1, -1e12)

        # Sigmoid激活函数输出实体概率
        entity_probabilities = torch.sigmoid(attention_scores)

        return entity_probabilities

class MedicalEntityExtractor(nn.Module):
    """CMeEE-V2嵌套实体识别网络"""
    def __init__(self, model_path, num_labels=15, head_size=64, use_rope=True):
        super(MedicalEntityExtractor, self).__init__()
        # 加载预训练语言模型
        self.text_encoder = AutoModel.from_pretrained(model_path)
        self.representation_dim = self.text_encoder.config.hidden_size

        # 全局指针网络层
        self.entity_pointer_network = OptimizedGlobalPointer(
            hidden_dimension=self.representation_dim,
            head_count=num_labels,
            head_dimension=head_size,
            enable_rope=use_rope
        )

    def forward(self, token_ids, attention_weights, segment_ids=None):
        # 文本编码过程
        if segment_ids is not None:
            encoder_outputs = self.text_encoder(token_ids, attention_mask=attention_weights, token_type_ids=segment_ids)
        else:
            encoder_outputs = self.text_encoder(token_ids, attention_mask=attention_weights)

        # 提取最终隐藏层表示
        final_hidden_states = encoder_outputs.last_hidden_state

        # 实体指针预测
        entity_scores = self.entity_pointer_network(final_hidden_states, attention_weights)

        return entity_scores

class MedicalEntityPredictor:
    """CMeEE-V2实体识别预测器"""
    def __init__(self, model_path, config):
        self.configuration = config
        self.entity_model = MedicalEntityExtractor(
            model_path=model_path,
            num_labels=config['num_labels'],
            head_size=config['head_size'],
            use_rope=config['use_rope']
        )
        self.text_tokenizer = AutoTokenizer.from_pretrained(model_path)

        # 配置计算设备
        self.computation_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.entity_model.to(self.computation_device)
        self.entity_model.eval()

    def load_weights(self, model_weight_path):
        """导入模型参数"""
        state_dict = torch.load(model_weight_path, map_location=self.computation_device)

        # 处理可能的状态字典不匹配问题
        model_state_dict = self.entity_model.state_dict()

        # 移除不匹配的键
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"跳过形状不匹配的参数: {k}, 期望: {model_state_dict[k].shape}, 实际: {v.shape}")
            else:
                print(f"跳过不存在的参数: {k}")

        # 加载过滤后的状态字典
        missing_keys, unexpected_keys = self.entity_model.load_state_dict(filtered_state_dict, strict=False)

        if missing_keys:
            print(f"缺失的参数: {missing_keys}")
        if unexpected_keys:
            print(f"意外的参数: {unexpected_keys}")

        print(f"成功加载 {len(filtered_state_dict)} 个参数")
        return self
        
    def predict(self, input_text, confidence_threshold=0.5):
        """对输入文本执行实体识别"""
        # 使用滑动窗口处理长文本
        max_window_size = self.configuration['max_len']
        window_step_size = self.configuration['stride']

        # 文本编码处理
        encoded_text = self.text_tokenizer(
            input_text,
            return_offsets_mapping=True,
            add_special_tokens=True,
            return_tensors='pt'
        )

        token_sequence = encoded_text['input_ids'][0]
        attention_pattern = encoded_text['attention_mask'][0]
        position_mapping = encoded_text['offset_mapping'][0].tolist()

        # 滑动窗口实体识别
        collected_predictions = []
        total_tokens = len(token_sequence)

        for window_start in range(0, total_tokens, window_step_size):
            window_end = min(window_start + max_window_size, total_tokens)
            if window_end == total_tokens:
                window_start = max(0, window_end - max_window_size)

            # 提取当前窗口数据
            window_tokens = token_sequence[window_start:window_end].unsqueeze(0).to(self.computation_device)
            window_attention = attention_pattern[window_start:window_end].unsqueeze(0).to(self.computation_device)

            # 模型推理
            with torch.no_grad():
                model_outputs = self.entity_model(window_tokens, window_attention)
                prediction_scores = model_outputs[0].cpu()  # [num_heads, window_len, window_len]

            # 解析实体预测结果
            window_entities = []
            current_window_size = window_end - window_start

            for entity_type_idx in range(prediction_scores.shape[0]):
                for entity_start_pos in range(current_window_size):
                    for entity_end_pos in range(entity_start_pos + 1, current_window_size):
                        if prediction_scores[entity_type_idx, entity_start_pos, entity_end_pos] > confidence_threshold:
                            # 映射回原始文本位置
                            original_start_pos = position_mapping[window_start + entity_start_pos][0]

                            # GlobalPointer的end_idx指向实体结束后的下一个token，需要调整
                            # 确保索引不越界
                            adjusted_end_token = min(window_start + entity_end_pos - 1, len(position_mapping) - 1)
                            original_end_pos = position_mapping[adjusted_end_token][1]

                            if original_start_pos < original_end_pos:  # 验证实体有效性
                                window_entities.append({
                                    'start_idx': original_start_pos,
                                    'end_idx': original_end_pos,
                                    'label_idx': entity_type_idx
                                })

            collected_predictions.extend(window_entities)

        # 消除重复预测结果
        filtered_predictions = self._remove_duplicate_predictions(collected_predictions)

        return filtered_predictions
        
    def _remove_duplicate_predictions(self, predictions):
        """过滤重复的实体预测"""
        observed_entities = set()
        unique_entities = []

        for prediction in predictions:
            entity_signature = (prediction['start_idx'], prediction['end_idx'], prediction['label_idx'])
            if entity_signature not in observed_entities:
                observed_entities.add(entity_signature)
                unique_entities.append(prediction)

        return unique_entities
