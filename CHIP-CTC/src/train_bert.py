#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIP-CTC任务2：BERT模型训练脚本
训练基于BERT的临床试验筛选标准分类模型
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
    parser = argparse.ArgumentParser(description='训练BERT分类模型')

    # 数据参数
    parser.add_argument('--data_dir', type=str, default='../data/CHIP-CTC/CHIP-CTC',
                       help='数据目录路径')
    parser.add_argument('--train_file', type=str, default='CHIP-CTC_train.json',
                       help='训练数据文件名')
    parser.add_argument('--dev_file', type=str, default='CHIP-CTC_dev.json',
                       help='验证数据文件名')

    # 模型参数
    parser.add_argument('--model_name', type=str, default='bert-base-chinese',
                       choices=['bert-base-chinese', 'hfl/chinese-roberta-wwm-ext',
                               'hfl/chinese-macbert-base', 'hfl/chinese-bert-wwm-ext'],
                       help='预训练模型名称')
    parser.add_argument('--local_model_path', type=str, default=None,
                       help='本地模型路径（如果提供，将使用本地模型）')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度')

    # 训练参数
    parser.add_argument('--epochs', type=int, default=5,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='warmup步数')
    parser.add_argument('--use_class_weights', action='store_true',
                       help='是否使用类别权重')

    # 输出参数
    parser.add_argument('--output_dir', type=str, default='checkpoints/bert',
                       help='模型输出目录')
    parser.add_argument('--model_name_suffix', type=str, default='',
                       help='模型名称后缀')

    return parser.parse_args()

def main():
    """主函数"""
    args = parse_args()

    logger.info("="*60)
    logger.info("CHIP-CTC任务2：BERT模型训练")
    logger.info("="*60)

    # 打印参数
    logger.info("训练参数:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 数据加载和处理
    logger.info("\n1. 数据加载和处理...")
    data_processor = CHIPCTCDataProcessor(args.data_dir)

    # 加载训练数据
    train_data = data_processor.load_data(args.train_file)
    if not train_data:
        logger.error("无法加载训练数据")
        return

    # 加载验证数据
    dev_data = data_processor.load_data(args.dev_file)
    if not dev_data:
        logger.warning("无法加载验证数据，将从训练集中分割")
        train_data, dev_data = data_processor.split_data(train_data, test_size=0.1)

    # 数据分析
    train_analysis = data_processor.analyze_data(train_data, "训练集")
    dev_analysis = data_processor.analyze_data(dev_data, "验证集")

    # 准备数据
    train_dataset, category_map = data_processor.prepare_data(train_data, is_train=True)
    dev_dataset, _ = data_processor.prepare_data(dev_data, is_train=True)

    # 获取类别权重
    class_weights = None
    if args.use_class_weights:
        class_weights = data_processor.get_class_weights(train_dataset['labels'])
        logger.info("使用类别权重进行训练")

    # 2. 模型初始化
    logger.info("\n2. 模型初始化...")
    num_labels = len(category_map)
    logger.info(f"类别数: {num_labels}")

    bert_classifier = BERTClassifier(
        model_name=args.model_name,
        num_labels=num_labels,
        local_model_path=args.local_model_path
    )

    # 显示模型信息
    model_info = bert_classifier.get_model_info()
    logger.info(f"模型信息: {model_info}")

    # 3. 训练模型
    logger.info("\n3. 开始训练...")

    # 构建模型保存路径
    model_suffix = f"_{args.model_name_suffix}" if args.model_name_suffix else ""
    model_save_path = output_dir / f"bert_{args.model_name.replace('/', '_')}{model_suffix}"

    bert_classifier.train(
        train_texts=train_dataset['texts'],
        train_labels=train_dataset['labels'],
        val_texts=dev_dataset['texts'],
        val_labels=dev_dataset['labels'],
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        class_weights=class_weights,
        save_path=str(model_save_path)
    )

    # 4. 保存训练信息
    logger.info("\n4. 保存训练信息...")

    # 保存类别映射
    category_file = output_dir / "category_mapping.json"
    with open(category_file, 'w', encoding='utf-8') as f:
        json.dump({
            'category_map': category_map,
            'id2label': {str(v): k for k, v in category_map.items()},
            'num_labels': num_labels,
            'model_name': args.model_name,
            'max_length': args.max_length
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"类别映射已保存到: {category_file}")

    # 保存训练参数
    train_config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'warmup_steps': args.warmup_steps,
        'use_class_weights': args.use_class_weights,
        'train_samples': len(train_dataset['texts']),
        'dev_samples': len(dev_dataset['texts']),
        'num_labels': num_labels
    }

    config_file = output_dir / "train_config.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(train_config, f, ensure_ascii=False, indent=2)

    logger.info(f"训练配置已保存到: {config_file}")

    # 5. 最终评估
    logger.info("\n5. 最终模型评估...")

    # 重新加载最佳模型进行最终评估
    try:
        bert_classifier.load_model(str(model_save_path))

        # 在验证集上评估
        val_predictions, _ = bert_classifier.predict(dev_dataset['texts'], batch_size=args.batch_size)

        # 保存预测结果用于分析
        predictions_file = output_dir / "val_predictions.json"
        data_processor.save_predictions(val_predictions, dev_dataset['ids'], str(predictions_file))

        logger.info("训练完成！")
        logger.info(f"模型已保存到: {model_save_path}")
        logger.info(f"验证预测结果: {predictions_file}")

    except Exception as e:
        logger.error(f"最终评估失败: {e}")

if __name__ == "__main__":
    main()
