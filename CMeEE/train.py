import torch
import torch.nn as nn
import torch.optim as optim
import json
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from models.medical_entity_model import MedicalEntityExtractor

# 模型训练配置设置
TRAIN_SETTINGS = {
    'pretrained_model': r'D:\work\TextAM_keshe4\download_models\chinese-macbert-large',  # 使用本地下载的模型路径
    'dataset_path': r'../data/CMeEE-V2/CMeEE-V2',
    'checkpoint_save_path': './model_checkpoints',
    'sequence_max_length': 512,#序列最大长度
    'window_overlap': 256,#窗口重叠大小
    'training_batch_size': 4,#训练批次大小
    'total_epochs': 5,#总训练轮数
    'optimizer_lr': 2e-5,#优化器学习率
    'entity_categories': 9,#实体类别数
    'attention_head_dim': 64,#注意力头维度
    'enable_position_encoding': True,#是否启用位置编码
    'prediction_confidence': 0.5,#预测置信度
    'stop_training_threshold': 3,  # 当连续3轮没有改善时提前停止训练
    'min_improvement_delta': 1e-6  # 判定为改善的最小变化量
}

# 实体类型编码表 - 基于CMeEE-V2数据集标准
ENTITY_TYPE_MAPPING = {
    'dis': 0,  # 疾病相关
    'sym': 1,  # 症状表现
    'pro': 2,  # 诊疗流程
    'equ': 3,  # 医疗器械
    'dru': 4,  # 药物信息
    'ite': 5,  # 检测项目
    'bod': 6,  # 身体部位
    'dep': 7,  # 医疗机构
    'mic': 8   # 微生物种类
}

# 逆向实体映射表
LABEL_TO_ENTITY = {index: entity for entity, index in ENTITY_TYPE_MAPPING.items()}

def read_json_dataset(file_path):
    """读取JSON格式的数据集文件"""
    with open(file_path, 'r', encoding='utf-8') as file_handle:
        dataset_content = json.load(file_handle)
    return dataset_content

def generate_text_segments(text_content, tokenizer_model, segment_length=512, overlap_size=256):
    """生成文本分段以处理长文本"""
    # 使用tokenizer编码文本并获取位置映射
    encoded_result = tokenizer_model(
        text_content,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=False
    )

    token_ids = encoded_result['input_ids']
    attention_weights = encoded_result['attention_mask']
    position_map = encoded_result['offset_mapping']

    segments_list = []
    total_tokens = len(token_ids)

    for segment_start in range(0, total_tokens, overlap_size):
        segment_end = min(segment_start + segment_length, total_tokens)
        if segment_end == total_tokens:
            segment_start = max(0, segment_end - segment_length)

        segment_data = {
            'input_ids': token_ids[segment_start:segment_end],
            'attention_mask': attention_weights[segment_start:segment_end],
            'offset_mapping': position_map[segment_start:segment_end],
            'global_start': segment_start
        }
        segments_list.append(segment_data)

    return segments_list

def build_training_labels(entity_list, segment_info, tokenizer_model, entity_mapping):
    """构建训练用的标签矩阵"""
    token_sequence = segment_info['input_ids']
    position_offsets = segment_info['offset_mapping']
    segment_offset = segment_info['global_start']

    # 初始化标签矩阵 [类别数, 窗口长度, 窗口长度]
    total_categories = len(entity_mapping)
    segment_size = len(token_sequence)
    label_matrix = torch.zeros((total_categories, segment_size, segment_size))

    for entity_info in entity_list:
        entity_begin = entity_info['start_idx']
        entity_finish = entity_info['end_idx']
        entity_category = entity_info['type']

        if entity_category not in entity_mapping:
            continue

        category_index = entity_mapping[entity_category]

        # 确定实体在当前段落中的token边界
        token_begin_pos = None
        token_end_pos = None

        for token_idx in range(segment_size):
            token_range = position_offsets[token_idx]
            global_token_idx = segment_offset + token_idx

            # 检测token是否包含实体的开始位置
            if token_range[0] <= entity_begin < token_range[1]:
                token_begin_pos = token_idx

            # 检测token是否包含实体的结束位置
            if token_range[0] < entity_finish <= token_range[1]:
                token_end_pos = token_idx + 1  # 结束位置指向下一个token

        # 验证边界并修正范围
        if token_begin_pos is not None and token_end_pos is not None:
            # 确保结束位置不超过段落长度（左闭右开区间）
            token_end_pos = min(token_end_pos, segment_size - 1)

            if token_begin_pos < token_end_pos:
                # 在标签矩阵中标记实体位置
                label_matrix[category_index, token_begin_pos, token_end_pos] = 1.0

    return label_matrix

def batch_data_processor(batch_data):
    """处理批次数据的填充和对齐"""
    token_sequences = [sample['input_ids'] for sample in batch_data]
    attention_masks = [sample['attention_mask'] for sample in batch_data]
    target_labels = [sample['labels'] for sample in batch_data]

    # 计算批次中的最大序列长度
    max_sequence_len = max(len(seq) for seq in token_sequences)

    # 对每个样本进行填充处理
    for sample_idx in range(len(token_sequences)):
        padding_needed = max_sequence_len - len(token_sequences[sample_idx])
        # 填充token序列
        token_sequences[sample_idx] += [0] * padding_needed
        # 填充注意力掩码
        attention_masks[sample_idx] += [0] * padding_needed

        # 扩展标签矩阵维度
        original_matrix = target_labels[sample_idx]
        categories_count, original_len, _ = original_matrix.shape
        expanded_matrix = torch.zeros((categories_count, max_sequence_len, max_sequence_len))
        expanded_matrix[:, :original_len, :original_len] = original_matrix
        target_labels[sample_idx] = expanded_matrix

    # 转换为PyTorch张量
    token_tensor = torch.tensor(token_sequences)
    mask_tensor = torch.tensor(attention_masks)
    label_tensor = torch.stack(target_labels)

    return {
        'input_ids': token_tensor,
        'attention_mask': mask_tensor,
        'labels': label_tensor
    }

def execute_training_epoch(model_network, data_loader, optimizer_tool, loss_calculator, compute_device):
    """执行一轮完整的训练过程"""
    model_network.train()
    accumulated_loss = 0.0

    # 使用进度条显示训练进度
    progress_bar = tqdm(data_loader, desc="Training", unit="batch", ncols=100)

    for batch_samples in progress_bar:
        # 将数据移动到计算设备
        token_inputs = batch_samples['input_ids'].to(compute_device)
        attention_weights = batch_samples['attention_mask'].to(compute_device)
        ground_truth = batch_samples['labels'].to(compute_device)

        # 重置优化器梯度
        optimizer_tool.zero_grad()

        # 模型前向计算
        model_predictions = model_network(token_inputs, attention_weights)

        # 计算当前批次的损失
        current_loss = loss_calculator(model_predictions, ground_truth)
        accumulated_loss += current_loss.item()

        # 执行反向传播和参数更新
        current_loss.backward()
        optimizer_tool.step()

        # 更新进度条显示当前损失
        progress_bar.set_postfix(loss=f"{current_loss.item():.4f}")

    average_loss = accumulated_loss / len(data_loader)
    progress_bar.close()
    return average_loss

def perform_model_validation(model_network, validation_loader, loss_calculator, compute_device):
    """对模型进行验证评估"""
    model_network.eval()
    total_validation_loss = 0.0

    # 使用进度条显示验证进度
    progress_bar = tqdm(validation_loader, desc="Evaluating", unit="batch", ncols=100)

    with torch.no_grad():
        for batch_samples in progress_bar:
            # 数据迁移到设备
            token_inputs = batch_samples['input_ids'].to(compute_device)
            attention_weights = batch_samples['attention_mask'].to(compute_device)
            ground_truth = batch_samples['labels'].to(compute_device)

            # 模型推理
            model_outputs = model_network(token_inputs, attention_weights)

            # 累积损失
            batch_loss = loss_calculator(model_outputs, ground_truth)
            total_validation_loss += batch_loss.item()

            # 更新进度条
            progress_bar.set_postfix(loss=f"{batch_loss.item():.4f}")

    average_validation_loss = total_validation_loss / len(validation_loader)
    progress_bar.close()
    return average_validation_loss

def run_training_pipeline():
    """执行完整的模型训练流程"""
    print("=== 任务1: 医学实体抽取 - 开始模型训练 ===")

    # 第一阶段：数据读取
    print("#######读取数据集")
    training_dataset_path = os.path.join(TRAIN_SETTINGS['dataset_path'], 'CMeEE-V2_train.json')
    validation_dataset_path = os.path.join(TRAIN_SETTINGS['dataset_path'], 'CMeEE-V2_dev.json')

    training_dataset = read_json_dataset(training_dataset_path)
    validation_dataset = read_json_dataset(validation_dataset_path)

    print(f"训练数据集规模: {len(training_dataset)}")
    print(f"验证数据集规模: {len(validation_dataset)}")

    # 第二阶段：准备tokenizer
    print("#######初始化文本处理器")
    text_tokenizer = AutoTokenizer.from_pretrained(TRAIN_SETTINGS['pretrained_model'])

    # 第三阶段：构建训练样本
    print("#######构建训练样本")
    training_samples = []

    for data_item in training_dataset:
        content_text = data_item['text']
        annotated_entities = data_item['entities']

        # 生成文本分段
        text_segments = generate_text_segments(
            content_text,
            text_tokenizer,
            TRAIN_SETTINGS['sequence_max_length'],
            TRAIN_SETTINGS['window_overlap']
        )

        for segment in text_segments:
            # 生成对应的标签矩阵
            segment_labels = build_training_labels(
                annotated_entities,
                segment,
                text_tokenizer,
                ENTITY_TYPE_MAPPING
            )

            training_sample = {
                'input_ids': segment['input_ids'],
                'attention_mask': segment['attention_mask'],
                'labels': segment_labels
            }
            training_samples.append(training_sample)

    print(f"训练样本总量: {len(training_samples)}")
    
    # 第四阶段：构建验证样本
    print("#######构建验证样本")
    validation_samples = []

    for data_item in validation_dataset:
        content_text = data_item['text']
        annotated_entities = data_item['entities']

        # 生成文本分段
        text_segments = generate_text_segments(
            content_text,
            text_tokenizer,
            TRAIN_SETTINGS['sequence_max_length'],
            TRAIN_SETTINGS['window_overlap']
        )

        for segment in text_segments:
            # 生成标签矩阵
            segment_labels = build_training_labels(
                annotated_entities,
                segment,
                text_tokenizer,
                ENTITY_TYPE_MAPPING
            )

            validation_sample = {
                'input_ids': segment['input_ids'],
                'attention_mask': segment['attention_mask'],
                'labels': segment_labels
            }
            validation_samples.append(validation_sample)

    print(f"验证样本总量: {len(validation_samples)}")

    # 第五阶段：设置数据加载器
    print("#######配置数据加载器")
    training_data_loader = torch.utils.data.DataLoader(
        training_samples,
        batch_size=TRAIN_SETTINGS['training_batch_size'],
        shuffle=True,
        collate_fn=batch_data_processor
    )

    validation_data_loader = torch.utils.data.DataLoader(
        validation_samples,
        batch_size=TRAIN_SETTINGS['training_batch_size'],
        shuffle=False,
        collate_fn=batch_data_processor
    )

    # 第六阶段：模型初始化
    print("#######构建模型架构")
    compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {compute_device}")

    entity_model = MedicalEntityExtractor(
        model_path=TRAIN_SETTINGS['pretrained_model'],
        num_labels=TRAIN_SETTINGS['entity_categories'],
        head_size=TRAIN_SETTINGS['attention_head_dim'],
        use_rope=TRAIN_SETTINGS['enable_position_encoding']
    )
    entity_model.to(compute_device)

    # 第七阶段：配置优化策略
    print("#######设置优化器和损失计算")
    parameter_optimizer = optim.AdamW(entity_model.parameters(), lr=TRAIN_SETTINGS['optimizer_lr'])
    loss_function = nn.BCELoss()  # 二元交叉熵损失
    
    # 第八阶段：准备模型存储目录
    if not os.path.exists(TRAIN_SETTINGS['checkpoint_save_path']):
        os.makedirs(TRAIN_SETTINGS['checkpoint_save_path'])

    # 第九阶段：执行模型训练
    print("#######开始训练")
    best_loss = float('inf')

    # 早停机制参数
    stagnation_count = 0
    max_stagnation_epochs = TRAIN_SETTINGS['stop_training_threshold']
    minimum_improvement = TRAIN_SETTINGS['min_improvement_delta']

    for epoch_idx in range(TRAIN_SETTINGS['total_epochs']):
        print(f"\nEpoch {epoch_idx+1}/{TRAIN_SETTINGS['total_epochs']}")

        # 执行训练阶段
        training_epoch_loss = execute_training_epoch(
            entity_model,
            training_data_loader,
            parameter_optimizer,
            loss_function,
            compute_device
        )

        # 执行验证阶段
        validation_epoch_loss = perform_model_validation(
            entity_model,
            validation_data_loader,
            loss_function,
            compute_device
        )

        print(f"Train Loss: {training_epoch_loss:.4f}")
        print(f"Dev Loss: {validation_epoch_loss:.4f}")

        # 模型保存策略
        if validation_epoch_loss < best_loss - minimum_improvement:
            best_loss = validation_epoch_loss
            best_model_file = os.path.join(TRAIN_SETTINGS['checkpoint_save_path'], 'best_model.pth')
            torch.save(entity_model.state_dict(), best_model_file)
            print(f"保存最佳模型到: {best_model_file}")

            # 重置早停计数
            stagnation_count = 0
        else:
            # 增加早停计数
            stagnation_count += 1
            print(f"早停计数器: {stagnation_count}/{max_stagnation_epochs}")

            # 触发早停条件
            if stagnation_count >= max_stagnation_epochs:
                print(f"\n早停触发! 连续 {max_stagnation_epochs} 个epoch验证损失没有改进。")
                break

    # 第十阶段：保存最终模型状态
    final_model_file = os.path.join(TRAIN_SETTINGS['checkpoint_save_path'], 'final_model.pth')
    torch.save(entity_model.state_dict(), final_model_file)
    print(f"保存最终模型至: {final_model_file}")

    # 第十一阶段：导出训练配置
    config_export_file = os.path.join(TRAIN_SETTINGS['checkpoint_save_path'], 'training_config.json')
    with open(config_export_file, 'w', encoding='utf-8') as config_file:
        json.dump(TRAIN_SETTINGS, config_file, ensure_ascii=False, indent=2)
    print(f"导出配置参数至: {config_export_file}")

    print("\n=== 训练流程执行完毕 ===")


if __name__ == '__main__':
    run_training_pipeline()
