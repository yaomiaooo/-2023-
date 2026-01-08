from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent  # web/backend -> web -> TextAM_keshe4

# 添加所有必要的路径
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'CMeEE'))
sys.path.append(str(project_root / 'CHIP-CTC'))
sys.path.append(str(project_root / 'CMeEE' / 'models'))
sys.path.append(str(project_root / 'CHIP-CTC' / 'models'))
sys.path.append(str(project_root / 'CHIP-CTC' / 'utils'))

# 导入任务1的模型
try:
    from CMeEE.models.medical_entity_model import MedicalEntityPredictor
except ImportError:
    # 如果上面的导入失败，尝试直接导入
    from medical_entity_model import MedicalEntityPredictor

# 导入任务2的模型和数据处理器
try:
    from CHIP_CTC.models.bert_classifier import BERTClassifier
    from CHIP_CTC.utils.data_processor import CHIPCTCDataProcessor
except ImportError:
    # 如果上面的导入失败，尝试直接导入
    from bert_classifier import BERTClassifier
    from data_processor import CHIPCTCDataProcessor

app = Flask(__name__)
CORS(app)

# 全局变量存储模型
models = {}

def load_models():
    """加载训练好的模型"""

    # 任务1：医学实体抽取模型
    print("加载任务1：医学实体抽取模型...")
    task1_config = {
        'max_len': 512,
        'stride': 256,
        'num_labels': 9,
        'head_size': 64,
        'use_rope': True
    }

    task1_model_path = str(project_root / 'download_models' / 'chinese-macbert-large')
    task1_predictor = MedicalEntityPredictor(task1_model_path, task1_config)

    # 加载训练好的权重
    model_weights_path = r'D:\work\TextAM_keshe4\CMeEE\model_checkpoints\best_model.pth'
    task1_predictor.load_weights(model_weights_path)

    models['task1'] = task1_predictor
    print("任务1模型加载完成")

    # 任务2：临床试验筛选标准分类模型
    print("加载任务2：文本分类模型...")
    task2_model_path = r'D:\work\TextAM_keshe4\CHIP-CTC\checkpoints\bert\bert_bert-base-chinese_base'
    task2_classifier = BERTClassifier(local_model_path=task2_model_path, num_labels=44)
    task2_classifier.load_model(task2_model_path)

    # 加载数据处理器和类别映射
    data_processor = CHIPCTCDataProcessor(r'D:\work\TextAM_keshe4\data\CHIP-CTC\CHIP-CTC')
    category_mapping_file = r'D:\work\TextAM_keshe4\CHIP-CTC\checkpoints\bert\category_mapping.json'
    with open(category_mapping_file, 'r', encoding='utf-8') as f:
        category_data = json.load(f)
    data_processor.category_map = category_data.get('category_map', {})

    if data_processor.category_map:
        sorted_classes = sorted(data_processor.category_map.keys(),
                              key=lambda x: data_processor.category_map[x])
        data_processor.label_encoder.fit(sorted_classes)

    models['task2'] = {
        'classifier': task2_classifier,
        'data_processor': data_processor
    }
    print("任务2模型加载完成")

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys())
    })

@app.route('/api/task1/predict', methods=['POST'])
def predict_task1():
    """任务1：医学实体抽取预测"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '缺少文本参数'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': '文本不能为空'}), 400

        if 'task1' not in models:
            return jsonify({'error': '任务1模型未加载'}), 500

        predictor = models['task1']

        # 执行预测
        predictions = predictor.predict(text)

        # 格式化结果
        entities = []
        for pred in predictions:
            entity_text = text[pred['start_idx']:pred['end_idx']]
            entity_type_map = {
                0: '疾病(dis)',
                1: '症状(sym)',
                2: '医疗程序(pro)',
                3: '医疗设备(equ)',
                4: '药物(dru)',
                5: '医学检验项目(ite)',
                6: '身体(bod)',
                7: '科室(dep)',
                8: '微生物类(mic)'
            }

            entities.append({
                'text': entity_text,
                'type': entity_type_map.get(pred['label_idx'], '未知'),
                'start': pred['start_idx'],
                'end': pred['end_idx']
            })

        return jsonify({
            'success': True,
            'text': text,
            'entities': entities,
            'entity_count': len(entities)
        })

    except Exception as e:
        print(f"任务1预测错误: {str(e)}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/task2/predict', methods=['POST'])
def predict_task2():
    """任务2：临床试验筛选标准分类预测"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': '缺少文本参数'}), 400

        text = data['text'].strip()
        if not text:
            return jsonify({'error': '文本不能为空'}), 400

        if 'task2' not in models:
            return jsonify({'error': '任务2模型未加载'}), 500

        classifier = models['task2']['classifier']
        data_processor = models['task2']['data_processor']

        # 执行预测
        predictions, probabilities = classifier.predict([text])

        # 获取预测类别
        predicted_class_idx = predictions[0]
        predicted_class = data_processor.label_encoder.inverse_transform([predicted_class_idx])[0]

        # 获取概率分布（前5个最可能的类别）
        probs = probabilities[0]
        top_indices = probs.argsort()[-5:][::-1]
        top_probabilities = []

        for idx in top_indices:
            class_name = data_processor.label_encoder.inverse_transform([idx])[0]
            probability = float(probs[idx])
            top_probabilities.append({
                'class': class_name,
                'probability': probability
            })

        return jsonify({
            'success': True,
            'text': text,
            'prediction': predicted_class,
            'top_probabilities': top_probabilities
        })

    except Exception as e:
        print(f"任务2预测错误: {str(e)}")
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/api/task1/examples', methods=['GET'])
def get_task1_examples():
    """获取任务1示例数据"""
    examples = [
        "患者出现发热、咳嗽、呼吸困难等症状，胸部CT显示双肺多发磨玻璃密度影，符合新型冠状病毒肺炎表现。",
        "患儿因反复发热、皮疹就诊，实验室检查提示血常规正常，病毒抗体检测阳性，考虑为风疹病毒感染。",
        "患者糖尿病史10年，口服二甲双胍治疗，近期血糖控制不佳，建议调整治疗方案。",
        "心脏超声显示左心室收缩功能减低，EF值为45%，诊断为心力衰竭。"
    ]

    return jsonify({
        'examples': examples
    })

@app.route('/api/task2/examples', methods=['GET'])
def get_task2_examples():
    """获取任务2示例数据"""
    examples = [
        "年龄≥18岁，≤75岁",
        "体质量指数(BMI)≥18.5 kg/m²，≤30 kg/m²",
        "肝功能正常，无活动性肝炎或肝硬化",
        "肾功能正常，血肌酐清除率≥60 mL/min",
        "无严重心血管疾病史，如心肌梗死、心力衰竭等"
    ]

    return jsonify({
        'examples': examples
    })

@app.route('/api/dataset/stats/cmeee', methods=['GET'])
def get_cmeee_stats():
    """获取CMeEE数据集统计信息"""
    try:
        import json
        from collections import Counter, defaultdict

        # 读取训练数据集
        train_file = project_root / 'data' / 'CMeEE-V2' / 'CMeEE-V2' / 'CMeEE-V2_train.json'
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        # 基本统计
        total_samples = len(train_data)

        # 实体统计
        entity_counts = Counter()
        entity_types = Counter()
        text_lengths = []
        entities_per_sample = []

        for sample in train_data:
            text_lengths.append(len(sample['text']))
            entities_per_sample.append(len(sample['entities']))

            for entity in sample['entities']:
                entity_types[entity['type']] += 1
                # 统计实体长度
                entity_length = entity['end_idx'] - entity['start_idx']
                entity_counts[entity_length] += 1

        # 计算统计指标
        stats = {
            'dataset_name': 'CMeEE-V2',
            'description': '中文医学实体识别数据集',
            'basic_stats': {
                'total_samples': total_samples,
                'avg_text_length': round(sum(text_lengths) / len(text_lengths), 2),
                'max_text_length': max(text_lengths),
                'min_text_length': min(text_lengths),
                'avg_entities_per_sample': round(sum(entities_per_sample) / len(entities_per_sample), 2),
                'total_entities': sum(entity_types.values())
            },
            'entity_type_distribution': dict(entity_types),
            'text_length_distribution': {
                'bins': [0, 50, 100, 150, 200, 300, 500, 1000],
                'counts': [
                    len([l for l in text_lengths if l <= 50]),
                    len([l for l in text_lengths if 50 < l <= 100]),
                    len([l for l in text_lengths if 100 < l <= 150]),
                    len([l for l in text_lengths if 150 < l <= 200]),
                    len([l for l in text_lengths if 200 < l <= 300]),
                    len([l for l in text_lengths if 300 < l <= 500]),
                    len([l for l in text_lengths if 500 < l <= 1000]),
                    len([l for l in text_lengths if l > 1000])
                ]
            },
            'entity_length_distribution': dict(entity_counts.most_common(10))
        }

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        print(f"CMeEE数据集统计错误: {str(e)}")
        return jsonify({'error': f'统计失败: {str(e)}'}), 500

@app.route('/api/dataset/stats/chip_ctc', methods=['GET'])
def get_chip_ctc_stats():
    """获取CHIP-CTC数据集统计信息"""
    try:
        import json
        from collections import Counter

        # 读取训练数据
        train_file = project_root / 'data' / 'CHIP-CTC' / 'CHIP-CTC' / 'CHIP-CTC_train.json'
        test_file = project_root / 'data' / 'CHIP-CTC' / 'CHIP-CTC' / 'CHIP-CTC_test.json'

        # 读取数据
        with open(train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)

        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        # 合并训练和测试数据进行统计
        combined_data = train_data + test_data

        # 基本统计
        total_samples = len(combined_data)

        # 类别分布
        category_counts = Counter([item['label'] for item in combined_data])
        train_categories = Counter([item['label'] for item in train_data])
        test_categories = Counter([item['label'] for item in test_data])

        # 文本长度统计
        text_lengths = [len(item['text']) for item in combined_data]

        # 计算统计指标
        stats = {
            'dataset_name': 'CHIP-CTC',
            'description': '临床试验筛选标准分类数据集',
            'basic_stats': {
                'total_samples': total_samples,
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'num_categories': len(category_counts),
                'avg_text_length': round(sum(text_lengths) / len(text_lengths), 2),
                'max_text_length': max(text_lengths),
                'min_text_length': min(text_lengths)
            },
            'category_distribution': {
                'overall': dict(category_counts),
                'train': dict(train_categories),
                'test': dict(test_categories)
            },
            'text_length_distribution': {
                'bins': [0, 10, 20, 30, 50, 100, 200, 500],
                'counts': [
                    len([l for l in text_lengths if l <= 10]),
                    len([l for l in text_lengths if 10 < l <= 20]),
                    len([l for l in text_lengths if 20 < l <= 30]),
                    len([l for l in text_lengths if 30 < l <= 50]),
                    len([l for l in text_lengths if 50 < l <= 100]),
                    len([l for l in text_lengths if 100 < l <= 200]),
                    len([l for l in text_lengths if 200 < l <= 500]),
                    len([l for l in text_lengths if l > 500])
                ]
            },
            'category_examples': dict(category_counts.most_common(10))
        }

        return jsonify({
            'success': True,
            'stats': stats
        })

    except Exception as e:
        print(f"CHIP-CTC数据集统计错误: {str(e)}")
        return jsonify({'error': f'统计失败: {str(e)}'}), 500

if __name__ == '__main__':
    print("初始化医学文本挖掘系统...")
    load_models()
    print("模型加载完成，开始启动服务...")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )