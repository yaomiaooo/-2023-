#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIP-CTC任务2：BERT模型预测脚本
使用训练好的BERT模型进行预测
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_processor import CHIPCTCDataProcessor
from models.bert_classifier import BERTClassifier

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用BERT模型进行预测')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../data/CHIP-CTC/CHIP-CTC',
                       help='数据目录路径')
    parser.add_argument('--test_file', type=str, default='CHIP-CTC_test.json',
                       help='测试数据文件名')

    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--config_file', type=str, default=None,
                       help='模型配置文件路径（如果为None，则自动寻找）')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度')

    # 预测参数
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='results',
                       help='输出目录')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件名（如果为None，则自动生成）')

    return parser.parse_args()

def load_model_config(config_file):
    """加载模型配置"""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"加载模型配置: {config_file}")
        return config
    except Exception as e:
        logger.warning(f"无法加载配置文件 {config_file}: {e}")
        return {}

def main():
    """主函数"""
    args = parse_args()

    logger.info("="*60)
    logger.info("CHIP-CTC任务2：BERT模型预测")
    logger.info("="*60)

    # 打印参数
    logger.info("预测参数:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 加载模型配置
    logger.info("\n1. 加载模型配置...")

    # 如果没有指定配置文件，尝试自动寻找
    if args.config_file is None:
        config_file = Path(args.model_path) / "train_config.json"
        if config_file.exists():
            args.config_file = str(config_file)

    config = {}
    if args.config_file and Path(args.config_file).exists():
        config = load_model_config(args.config_file)

    # 从配置中获取参数，如果没有则使用默认值
    model_name = config.get('model_name', 'bert-base-chinese')
    max_length = config.get('max_length', args.max_length)
    num_labels = config.get('num_labels', 44)

    logger.info(f"模型名称: {model_name}")
    logger.info(f"最大序列长度: {max_length}")
    logger.info(f"类别数: {num_labels}")

    # 2. 初始化模型
    logger.info("\n2. 初始化模型...")
    bert_classifier = BERTClassifier(
        model_name=model_name,
        num_labels=num_labels
    )

    # 加载训练好的模型
    try:
        bert_classifier.load_model(args.model_path)
        logger.info(f"成功加载模型: {args.model_path}")
    except Exception as e:
        logger.error(f"加载模型失败: {e}")
        return

    # 3. 数据加载和处理
    logger.info("\n3. 数据加载和处理...")
    data_processor = CHIPCTCDataProcessor(args.data_dir)

    # 加载测试数据
    test_data = data_processor.load_data(args.test_file)
    if not test_data:
        logger.error("无法加载测试数据")
        return

    # 数据分析
    test_analysis = data_processor.analyze_data(test_data, "测试集", is_test=True)

    # 准备数据
    test_dataset, _ = data_processor.prepare_data(test_data, is_train=False)

    # 加载类别映射
    # 首先在模型路径下寻找
    category_mapping_file = Path(args.model_path) / "category_mapping.json"
    if not category_mapping_file.exists():
        # 如果找不到，尝试在上级目录寻找（训练时保存的位置）
        parent_dir = Path(args.model_path).parent
        category_mapping_file = parent_dir / "category_mapping.json"

    if category_mapping_file.exists():
        try:
            with open(category_mapping_file, 'r', encoding='utf-8') as f:
                category_data = json.load(f)
            data_processor.category_map = category_data.get('category_map', {})

            # 正确恢复label_encoder状态
            if data_processor.category_map:
                # 获取排序后的类别名称列表
                sorted_classes = sorted(data_processor.category_map.keys(),
                                      key=lambda x: data_processor.category_map[x])
                # fit label_encoder
                data_processor.label_encoder.fit(sorted_classes)
                logger.info(f"成功加载类别映射，共 {len(sorted_classes)} 个类别")
            else:
                logger.warning("类别映射为空")
        except Exception as e:
            logger.warning(f"加载类别映射失败: {e}")
    else:
        logger.error(f"类别映射文件不存在: 尝试了 {args.model_path}/category_mapping.json 和 {Path(args.model_path).parent}/category_mapping.json")

    # 4. 进行预测
    logger.info("\n4. 开始预测...")
    predictions, probabilities = bert_classifier.predict(
        test_dataset['texts'],
        batch_size=args.batch_size
    )

    logger.info(f"预测完成，共处理 {len(predictions)} 条数据")

    # 5. 保存预测结果
    logger.info("\n5. 保存预测结果...")

    # 生成输出文件名
    if args.output_file is None:
        model_name_short = Path(args.model_path).name
        args.output_file = f"CHIP-CTC_test_pred_{model_name_short}.json"

    output_file = output_dir / args.output_file

    # 保存预测结果
    saved_results = data_processor.save_predictions(
        predictions,
        test_dataset['ids'],
        str(output_file),
        texts=test_dataset['texts']  # 添加文本参数
    )

    # 6. 保存概率文件（可选，用于集成学习）
    prob_file = output_dir / f"{output_file.stem}_probabilities.npy"
    try:
        import numpy as np
        np.save(prob_file, probabilities)
        logger.info(f"预测概率已保存到: {prob_file}")
    except Exception as e:
        logger.warning(f"保存概率文件失败: {e}")

    # 7. 统计预测结果
    logger.info("\n6. 预测结果统计...")

    # 预测类别分布
    from collections import Counter
    pred_labels = [data_processor.label_encoder.inverse_transform([p])[0] for p in predictions]
    pred_distribution = Counter(pred_labels)

    logger.info("预测类别分布 (前10个):")
    sorted_preds = sorted(pred_distribution.items(), key=lambda x: x[1], reverse=True)
    for label, count in sorted_preds[:10]:
        percentage = count / len(predictions) * 100
        logger.info(".1f")

    # 保存统计信息
    stats_file = output_dir / f"{output_file.stem}_stats.json"
    stats = {
        'total_predictions': len(predictions),
        'prediction_distribution': dict(pred_distribution),
        'model_path': args.model_path,
        'test_file': args.test_file,
        'output_file': str(output_file),
        'prediction_stats': {
            'unique_predictions': len(set(pred_labels)),
            'most_common': sorted_preds[0] if sorted_preds else None,
            'least_common': sorted_preds[-1] if sorted_preds else None
        }
    }

    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(f"预测统计已保存到: {stats_file}")

    logger.info("\n预测完成！")
    logger.info(f"最终结果文件: {output_file}")

if __name__ == "__main__":
    main()
