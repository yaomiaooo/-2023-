#CHIP-CTC任务2：主执行脚本


import os
import sys
import json
import argparse
import logging
from pathlib import Path
import subprocess

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """运行命令并返回结果"""
    logger.info(f"执行: {description}")
    logger.info(f"命令: {' '.join(cmd)}")

    try:
        # 首先尝试UTF-8编码
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        except UnicodeDecodeError:
            # 如果UTF-8失败，尝试GBK编码（Windows中文）
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, encoding='gbk')
            except UnicodeDecodeError:
                # 如果都失败，使用bytes模式并忽略错误
                result = subprocess.run(cmd, capture_output=True, text=False)
                # 手动解码，忽略错误
                result.stdout = result.stdout.decode('utf-8', errors='ignore') if result.stdout else ""
                result.stderr = result.stderr.decode('utf-8', errors='ignore') if result.stderr else ""

        if result.returncode == 0:
            logger.info(f"✓ {description} 成功完成")
            # 不打印详细输出，避免编码问题
            return True
        else:
            logger.error(f"✗ {description} 失败")
            if result.stderr:
                logger.error(f"错误:\n{result.stderr}")
            return False

    except Exception as e:
        logger.error(f"执行命令时出错: {e}")
        return False

def train_bert_models():
    """训练多个BERT模型"""
    logger.info("开始训练多个BERT模型...")

    models_to_train = [
        {
            'name': 'bert-base-chinese',
            'suffix': 'base',
            'local_path': None
        },
        {
            'name': 'hfl/chinese-roberta-wwm-ext',
            'suffix': 'roberta',
            'local_path': '../download_models/chinese-roberta-wwm-ext'  # 本地RoBERTa路径
        },
        {
            'name': 'hfl/chinese-macbert-base',
            'suffix': 'macbert',
            'local_path': '../download_models/chinese-macbert-large'
        }
    ]

    trained_models = []

    for model_config in models_to_train:
        logger.info(f"训练模型: {model_config['name']}")

        # 检查本地路径是否存在
        local_path = model_config.get('local_path')
        if local_path:
            import os
            if os.path.exists(local_path):
                logger.info(f"✓ 找到本地模型: {local_path}")
            else:
                logger.warning(f"✗ 本地模型路径不存在: {local_path}，将从网络下载")

        cmd = [
            sys.executable, 'src/train_bert.py',
            '--model_name', model_config['name'],
            '--model_name_suffix', model_config['suffix'],
            '--epochs', '5',
            '--batch_size', '16',
            '--learning_rate', '2e-5',
            '--output_dir', 'checkpoints/bert'
        ]

        # 如果有本地路径且存在，添加到命令中
        if local_path and os.path.exists(local_path):
            cmd.extend(['--local_model_path', local_path])
            logger.info(f"使用本地模型路径: {local_path}")

        if run_command(cmd, f"训练{model_config['name']}"):
            model_path = f"checkpoints/bert/bert_{model_config['name'].replace('/', '_')}_{model_config['suffix']}"
            trained_models.append({
                'name': model_config['name'],
                'path': model_path,
                'suffix': model_config['suffix']
            })
        else:
            logger.warning(f"模型 {model_config['name']} 训练失败，跳过")

    return trained_models

def predict_with_bert_models(trained_models):
    """使用训练好的BERT模型进行预测"""
    logger.info("使用BERT模型进行预测...")

    prediction_results = []

    for model_info in trained_models:
        logger.info(f"使用模型进行预测: {model_info['name']}")

        cmd = [
            sys.executable, 'src/predict_bert.py',
            '--model_path', model_info['path'],
            '--output_dir', 'results/bert_predictions',
            '--output_file', f"CHIP-CTC_test_pred_{model_info['suffix']}.json"
        ]

        if run_command(cmd, f"使用{model_info['name']}进行预测"):
            result_file = f"results/bert_predictions/CHIP-CTC_test_pred_{model_info['suffix']}.json"
            prob_file = f"results/bert_predictions/CHIP-CTC_test_pred_{model_info['suffix']}_probabilities.npy"

            prediction_results.append({
                'model': model_info['name'],
                'result_file': result_file,
                'prob_file': prob_file,
                'suffix': model_info['suffix']
            })

    return prediction_results

def ensemble_predictions(prediction_results):
    """集成多个模型的预测结果"""
    logger.info("集成多个模型的预测结果...")

    try:
        import numpy as np
        from collections import Counter

        # 加载所有概率文件
        all_probabilities = []
        model_names = []

        for pred_result in prediction_results:
            prob_file = pred_result['prob_file']
            if os.path.exists(prob_file):
                probs = np.load(prob_file)
                all_probabilities.append(probs)
                model_names.append(pred_result['suffix'])
                logger.info(f"加载概率文件: {prob_file}, 形状: {probs.shape}")
            else:
                logger.warning(f"概率文件不存在: {prob_file}")

        if not all_probabilities:
            logger.error("没有找到任何概率文件")
            return False

        # 计算平均概率
        avg_probabilities = np.mean(all_probabilities, axis=0)
        logger.info(f"平均概率形状: {avg_probabilities.shape}")

        # 集成预测（加权平均）
        weights = {
            'base': 0.3,      # BERT-base
            'roberta': 0.4,   # RoBERTa-wwm-ext
            'macbert': 0.3    # MacBERT
        }

        weighted_probs = np.zeros_like(avg_probabilities)
        for i, model_name in enumerate(model_names):
            weight = weights.get(model_name, 1.0 / len(model_names))
            weighted_probs += all_probabilities[i] * weight
            logger.info(f"{model_name} 权重: {weight}")

        # 最终预测
        final_predictions = np.argmax(weighted_probs, axis=1)

        # 加载测试数据获取ID
        test_file = "../data/CHIP-CTC/CHIP-CTC/CHIP-CTC_test.json"
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)

        test_ids = [item['id'] for item in test_data]

        # 保存集成结果
        ensemble_result = []
        for i, (pred, item_id) in enumerate(zip(final_predictions, test_ids)):
            # 获取对应的文本
            text = test_data[i]['text'] if i < len(test_data) else ""

            ensemble_result.append({
                'id': item_id,
                'label': str(pred),  # 先保存为数字，后续需要转换为类别名称
                'text': text
            })

        # 保存到文件
        output_file = "results/CHIP-CTC_test_pred_ensemble.json"
        os.makedirs("results", exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(ensemble_result, f, ensure_ascii=False, indent=2)

        logger.info(f"集成预测结果已保存到: {output_file}")

        # 统计预测分布
        pred_counter = Counter(final_predictions)
        logger.info("集成预测类别分布:")
        for label, count in sorted(pred_counter.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  类别 {label}: {count}")

        return True

    except Exception as e:
        logger.error(f"集成预测失败: {e}")
        return False

def create_submission_file():
    """创建最终提交文件"""
    logger.info("创建最终提交文件...")

    try:
        # 这里应该根据实际情况调整标签映射
        # 由于没有完整的标签映射，这里只是示例

        ensemble_file = "results/CHIP-CTC_test_pred_ensemble.json"
        submission_file = "results/CHIP-CTC_test_pred_final.json"

        if os.path.exists(ensemble_file):
            with open(ensemble_file, 'r', encoding='utf-8') as f:
                ensemble_results = json.load(f)

            # 这里需要将数字标签转换为实际的类别名称
            # 示例：假设我们有类别映射（需要根据实际情况调整）
            sample_categories = [
                "Age", "Symptom", "Disease", "Drug", "Multiple", "Therapy or Surgery",
                "Gender", "Lab", "Sign", "Addictive Behavior", "Pregnancy-related",
                "Diet", "Multiple", "Exercise", "Healthy"
            ]

            final_results = []
            for item in ensemble_results:
                label_idx = int(item['label'])
                if label_idx < len(sample_categories):
                    label_name = sample_categories[label_idx]
                else:
                    label_name = f"Category_{label_idx}"

                final_results.append({
                    'id': item['id'],
                    'label': label_name
                })

            with open(submission_file, 'w', encoding='utf-8') as f:
                json.dump(final_results, f, ensure_ascii=False, indent=2)

            logger.info(f"最终提交文件已创建: {submission_file}")
            return True
        else:
            logger.error(f"集成结果文件不存在: {ensemble_file}")
            return False

    except Exception as e:
        logger.error(f"创建提交文件失败: {e}")
        return False

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CHIP-CTC任务2主执行脚本')

    parser.add_argument('--mode', type=str, choices=['train', 'predict', 'full'],
                       default='full', help='执行模式：train(训练), predict(预测), full(完整流程)')

    parser.add_argument('--skip_ensemble', action='store_true',
                       help='跳过模型集成步骤')

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    logger.info("="*80)
    logger.info("CHIP-CTC任务2：临床试验筛选标准短文本分类")
    logger.info("="*80)

    logger.info(f"执行模式: {args.mode}")

    if args.mode in ['train', 'full']:
        # 1. 训练BERT模型
        logger.info("\n" + "="*60)
        logger.info("阶段1：训练BERT模型")
        logger.info("="*60)

        trained_models = train_bert_models()

        if not trained_models:
            logger.error("没有成功训练任何模型")
            return

        logger.info(f"成功训练了 {len(trained_models)} 个模型")

    if args.mode in ['predict', 'full']:
        # 2. 使用训练好的模型进行预测
        logger.info("\n" + "="*60)
        logger.info("阶段2：模型预测")
        logger.info("="*60)

        if args.mode == 'predict':
            # 如果只进行预测，需要手动指定训练好的模型
            trained_models = [
                {'name': 'bert-base-chinese', 'path': 'checkpoints/bert/bert_bert-base-chinese_base', 'suffix': 'base'},
                {'name': 'hfl/chinese-roberta-wwm-ext', 'path': 'checkpoints/bert/bert_hfl_chinese-roberta-wwm-ext_roberta', 'suffix': 'roberta'},
                {'name': 'hfl/chinese-macbert-base', 'path': 'checkpoints/bert/bert_hfl_chinese-macbert-base_macbert', 'suffix': 'macbert'}
            ]

        prediction_results = predict_with_bert_models(trained_models)

        if not prediction_results:
            logger.error("没有成功完成任何预测")
            return

        logger.info(f"成功完成了 {len(prediction_results)} 个模型的预测")

        # 3. 模型集成
        if not args.skip_ensemble:
            logger.info("\n" + "="*60)
            logger.info("阶段3：模型集成")
            logger.info("="*60)

            if ensemble_predictions(prediction_results):
                create_submission_file()
            else:
                logger.warning("模型集成失败")

    logger.info("\n" + "="*80)
    logger.info("CHIP-CTC任务2执行完成！")
    logger.info("="*80)

if __name__ == "__main__":
    main()
