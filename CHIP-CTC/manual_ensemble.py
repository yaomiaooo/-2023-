import os
import sys
import json
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def load_predictions(result_dir, model_suffix):
    """加载模型预测结果"""
    base_name = f"CHIP-CTC_test_pred_{model_suffix}"

    # 查找预测文件
    json_file = None
    prob_file = None

    for file in os.listdir(result_dir):
        if file.startswith(base_name) and file.endswith('.json'):
            json_file = os.path.join(result_dir, file)
        elif file.startswith(base_name) and 'probabilities' in file and file.endswith('.npy'):
            prob_file = os.path.join(result_dir, file)

    if not json_file or not prob_file:
        print(f"警告: {model_suffix} 的预测文件不完整")
        return None, None

    print(f"加载 {model_suffix} 预测结果: {json_file}")
    print(f"加载 {model_suffix} 概率文件: {prob_file}")

    # 加载概率
    probabilities = np.load(prob_file)

    return json_file, probabilities

def ensemble_predictions():
    """集成多个模型的预测结果"""
    print("开始集成预测...")

    # 模型配置
    models = [
        {
            'name': 'bert-base-chinese',
            'suffix': 'base',  
            'weight': 0.9,  # BERT权重90%
            'result_dir': 'results/bert_predictions'
        },
        {
            'name': 'hfl/chinese-roberta-wwm-ext',
            'suffix': 'roberta',  
            'weight': 0.1,  # RoBERTa权重10%
            'result_dir': 'results/bert_predictions'
        }
    ]

    # 加载所有模型的预测概率
    all_probabilities = []
    valid_models = []

    for model in models:
        json_file, probabilities = load_predictions(model['result_dir'], model['suffix'])
        if probabilities is not None:
            all_probabilities.append(probabilities * model['weight'])
            valid_models.append(model)
            print(f"✓ {model['name']}: {probabilities.shape}, 权重: {model['weight']}")
        else:
            print(f"✗ {model['name']}: 预测文件不存在")

    if not all_probabilities:
        print("错误: 没有找到任何有效的预测结果")
        return

    # 计算加权平均概率
    print("计算集成概率...")
    ensemble_probabilities = np.sum(all_probabilities, axis=0)

    # 归一化
    ensemble_probabilities = ensemble_probabilities / np.sum(ensemble_probabilities, axis=1, keepdims=True)

    # 得到最终预测
    final_predictions = np.argmax(ensemble_probabilities, axis=1)

    print(f"集成完成: {final_predictions.shape[0]} 条预测")

    # 加载测试数据获取文本
    test_file = "../data/CHIP-CTC/CHIP-CTC/CHIP-CTC_test.json"
    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 加载类别映射
    category_file = "checkpoints/bert/category_mapping.json"
    with open(category_file, 'r', encoding='utf-8') as f:
        category_data = json.load(f)

    id2label = category_data['id2label']

    # 加载第一个有效模型的结果来获取ID
    first_model = valid_models[0]
    with open(os.path.join(first_model['result_dir'], f"CHIP-CTC_test_pred_{first_model['suffix']}.json"), 'r', encoding='utf-8') as f:
        sample_results = json.load(f)

    # 创建集成结果
    ensemble_results = []
    for i, pred in enumerate(final_predictions):
        # 将预测的类别索引转换为类别名称
        pred_index = int(pred)
        pred_label = id2label.get(str(pred_index), f"Category_{pred_index}")

        # 获取对应的文本
        text = test_data[i]['text'] if i < len(test_data) else ""

        result_item = {
            'id': sample_results[i]['id'],
            'label': pred_label,
            'text': text
        }
        ensemble_results.append(result_item)

    # 保存集成结果
    output_file = "results/CHIP-CTC_test_pred_ensemble.json"
    os.makedirs("results", exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ensemble_results, f, ensure_ascii=False, indent=2)

    print(f"集成结果已保存: {output_file}")

    # 保存概率文件
    prob_file = "results/CHIP-CTC_test_pred_ensemble_probabilities.npy"
    np.save(prob_file, ensemble_probabilities)
    print(f"集成概率已保存: {prob_file}")

    # 统计信息
    pred_counts = {}
    for result in ensemble_results:
        label = result['label']
        pred_counts[label] = pred_counts.get(label, 0) + 1

    print("集成预测统计:")
    for label, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  类别 {label}: {count}")

    print("集成预测完成！")

if __name__ == "__main__":
    print("="*60)
    print("CHIP-CTC 手动集成预测")
    print("="*60)

    ensemble_predictions()

    print("="*60)
    print("集成完成，预测结果保存在 results/CHIP-CTC_test_pred_ensemble.json")
    print("="*60)
