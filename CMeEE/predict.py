import torch
import json
import os
from tqdm import tqdm
from models.medical_entity_model import MedicalEntityPredictor

# 预测系统配置参数
PREDICTION_CONFIG = {
    'pretrained_model_path': r'D:\work\TextAM_keshe4\download_models\chinese-macbert-large',  # 使用本地下载的模型路径
    'test_dataset_path': '../data/CMeEE-V2/CMeEE-V2',  # 修改为相对路径
    'model_checkpoint_dir': './model_checkpoints',
    'prediction_output_path': './submit/CMeEE-V2_pre.json',
    'max_sequence_length': 512,
    'sliding_window_stride': 256,
    'total_entity_types': 9,
    'attention_head_dimension': 64,
    'enable_rope_encoding': True,
    'confidence_threshold': 0.5,
    'max_prediction_samples': 3000  # 限制预测样本数量，用于快速测试
}

# 实体类别标签映射表 
ENTITY_TYPE_LOOKUP = {
    0: 'dis',  # 疾病
    1: 'sym',  # 临床表现
    2: 'pro',  # 医疗程序
    3: 'equ',  # 医疗设备
    4: 'dru',  # 药物
    5: 'ite',  # 医学检验项目
    6: 'bod',  # 身体
    7: 'dep',  # 科室
    8: 'mic'   # 微生物类
}

def load_json_data(file_path):
    """读取JSON格式的数据文件"""
    with open(file_path, 'r', encoding='utf-8') as file_stream:
        dataset = json.load(file_stream)
    return dataset

def extract_entities_from_text(entity_predictor, input_text, confidence_cutoff=0.5):
    """对单个文本执行实体识别"""
    # 调用预测器进行实体识别
    prediction_results = entity_predictor.predict(input_text, confidence_cutoff)

    # 格式化预测结果
    identified_entities = []
    for prediction in prediction_results:
        entity_start_pos = prediction['start_idx']
        entity_end_pos = prediction['end_idx']
        entity_category_idx = prediction['label_idx']

        extracted_text = input_text[entity_start_pos:entity_end_pos]
        entity_category = ENTITY_TYPE_LOOKUP[entity_category_idx]

        identified_entities.append({
            'start_idx': entity_start_pos,
            'end_idx': entity_end_pos,
            'type': entity_category,
            'entity': extracted_text
        })

    return identified_entities

def execute_prediction_pipeline():
    """执行完整的实体识别预测流程"""
    print("=== 任务1: 医学实体抽取 - 开始预测 ===")

    # 第一步：准备输出目录
    print("#######准备输出目录")
    os.makedirs(os.path.dirname(PREDICTION_CONFIG['prediction_output_path']), exist_ok=True)

    # 第二步：初始化预测模块
    print("#######初始化实体识别器")
    entity_recognizer = MedicalEntityPredictor(
        model_path=PREDICTION_CONFIG['pretrained_model_path'],
        config={
            'max_len': PREDICTION_CONFIG['max_sequence_length'],
            'stride': PREDICTION_CONFIG['sliding_window_stride'],
            'num_labels': PREDICTION_CONFIG['total_entity_types'],
            'head_size': PREDICTION_CONFIG['attention_head_dimension'],
            'use_rope': PREDICTION_CONFIG['enable_rope_encoding']
        }
    )

    # 第三步：加载训练好的模型参数
    model_checkpoint_path = os.path.join(PREDICTION_CONFIG['model_checkpoint_dir'], 'best_model.pth')
    entity_recognizer.load_weights(model_checkpoint_path)
    print(f"加载模型参数: {model_checkpoint_path}")

    # 第四步：读取测试数据集
    print("#######读取测试数据集")
    test_dataset_file = os.path.join(PREDICTION_CONFIG['test_dataset_path'], 'CMeEE-V2_test.json')
    test_dataset = load_json_data(test_dataset_file)
    print(f"测试数据集总规模: {len(test_dataset)}")

    # 限制预测样本数量，用于快速验证
    sample_limit = PREDICTION_CONFIG.get('max_prediction_samples')
    if sample_limit is not None and sample_limit > 0:
        test_dataset = test_dataset[:sample_limit]

    print(f"本次预测处理样本数: {len(test_dataset)}")

    # 第五步：执行批量预测
    print("#######开始实体识别")
    prediction_results = []

    # 使用进度条显示预测进度
    for test_sample in tqdm(test_dataset, desc="实体识别", unit="sample", ncols=100):
        sample_text = test_sample['text']

        # 执行实体识别
        detected_entities = extract_entities_from_text(
            entity_recognizer,
            sample_text,
            PREDICTION_CONFIG['confidence_threshold']
        )

        # 组织预测结果
        prediction_record = {
            'text': sample_text,
            'entities': detected_entities
        }

        prediction_results.append(prediction_record)

    # 第六步：导出预测结果
    print("#######导出识别结果")
    with open(PREDICTION_CONFIG['prediction_output_path'], 'w', encoding='utf-8') as output_file:
        json.dump(prediction_results, output_file, ensure_ascii=False, indent=2)

    print(f"结果文件保存至: {PREDICTION_CONFIG['prediction_output_path']}")
    print(f"预测任务完成，处理文本总数: {len(prediction_results)}")
    print("=== 预测完毕 ===")

if __name__ == '__main__':
    execute_prediction_pipeline()
