import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CHIPCTCDataProcessor:
    """CHIP-CTC临床试验筛选标准分类数据处理器"""

    def __init__(self, data_dir='../data/CHIP-CTC/CHIP-CTC'):
        self.data_dir = data_dir
        self.label_encoder = LabelEncoder()
        self.category_map = {}

        # 加载类别映射
        self._load_category_mapping()

    def _load_category_mapping(self):
        """加载类别映射"""
        try:
            category_file = f"{self.data_dir}/category.xlsx"
            df = pd.read_excel(category_file)
            # 假设第一列是类别名称
            categories = df.iloc[:, 0].tolist()
            self.category_map = {cat: i for i, cat in enumerate(categories)}
            logger.info(f"加载了 {len(categories)} 个类别")
        except Exception as e:
            logger.warning(f"无法加载类别文件: {e}")
            # 如果无法加载，使用训练数据中的类别
            pass

    def load_data(self, filename):
        """加载JSON格式的数据"""
        filepath = f"{self.data_dir}/{filename}"
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"从 {filepath} 加载了 {len(data)} 条数据")
            return data
        except Exception as e:
            logger.error(f"加载数据失败 {filepath}: {e}")
            return []

    def analyze_data(self, data, name="数据集", is_test=False):
        """分析数据集统计信息

        Args:
            data: 数据集
            name: 数据集名称
            is_test: 是否为测试集（测试集不分析标签）
        """
        if not data:
            return {}

        texts = [item['text'] for item in data]

        # 文本长度统计
        text_lengths = [len(text) for text in texts]
        avg_length = np.mean(text_lengths)
        max_length = np.max(text_lengths)
        min_length = np.min(text_lengths)

        analysis = {
            'name': name,
            'total_samples': len(data),
            'avg_text_length': avg_length,
            'max_text_length': max_length,
            'min_text_length': min_length
        }

        logger.info(f"=== {name} 分析结果 ===")
        logger.info(f"样本总数: {len(data)}")
        logger.info(f"平均文本长度: {avg_length:.1f}")
        logger.info(f"最长文本长度: {max_length}")
        logger.info(f"最短文本长度: {min_length}")

        # 只在非测试集时分析标签
        if not is_test:
            labels = [item['label'] for item in data]
            label_counts = Counter(labels)
            unique_labels = len(label_counts)

            analysis.update({
                'unique_labels': unique_labels,
                'label_distribution': dict(label_counts)
            })

            logger.info(f"唯一标签数: {unique_labels}")

            # 显示标签分布（前10个）
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            logger.info("标签分布 (前10个):")
            for label, count in sorted_labels[:10]:
                logger.info(f"  {label}: {count}")
        else:
            logger.info("跳过标签分析（测试集）")

        return analysis

    def prepare_data(self, data, is_train=True):
        """准备数据用于模型训练/预测"""
        if not data:
            return {}, []

        texts = [item['text'] for item in data]

        if is_train:
            labels = [item['label'] for item in data]

            # 标签编码
            if not hasattr(self.label_encoder, 'classes_') or len(self.label_encoder.classes_) == 0:
                encoded_labels = self.label_encoder.fit_transform(labels)
                self.category_map = {label: idx for idx, label in enumerate(self.label_encoder.classes_)}
            else:
                encoded_labels = self.label_encoder.transform(labels)

            # 构建类别映射的反向映射
            id2label = {idx: label for label, idx in self.category_map.items()}

            dataset = {
                'texts': texts,
                'labels': encoded_labels,
                'original_labels': labels,
                'ids': [item.get('id', f'item_{i}') for i, item in enumerate(data)]
            }

            logger.info(f"训练数据准备完成: {len(texts)} 样本, {len(self.category_map)} 类别")
            return dataset, self.category_map
        else:
            dataset = {
                'texts': texts,
                'ids': [item.get('id', f'item_{i}') for i, item in enumerate(data)]
            }
            logger.info(f"测试数据准备完成: {len(texts)} 样本")
            return dataset, []

    def get_class_weights(self, labels):
        """计算类别权重以处理不平衡"""
        label_counts = Counter(labels)
        total_samples = len(labels)
        num_classes = len(label_counts)

        # 计算每个类别的权重
        weights = {}
        for label, count in label_counts.items():
            weights[label] = total_samples / (num_classes * count)

        # 转换为numpy数组
        class_weights = np.array([weights[i] for i in range(num_classes)])

        logger.info(f"类别权重: {class_weights}")
        return class_weights

    def split_data(self, data, test_size=0.2, random_state=42):
        """分割数据为训练集和验证集"""
        train_data, val_data = train_test_split(
            data,
            test_size=test_size,
            random_state=random_state,
            stratify=[item['label'] for item in data]
        )

        logger.info(f"数据分割完成: 训练集 {len(train_data)}, 验证集 {len(val_data)}")
        return train_data, val_data

    def save_predictions(self, predictions, ids, output_file, texts=None):
        """保存预测结果

        Args:
            predictions: 预测结果
            ids: ID列表
            output_file: 输出文件路径
            texts: 可选的文本列表（用于测试集提交）
        """
        results = []
        for i, (pred, item_id) in enumerate(zip(predictions, ids)):
            # 如果pred是numpy数组，取第一个元素
            if isinstance(pred, np.ndarray):
                pred = pred.item() if pred.ndim == 0 else pred[0]

            # 转换回原始标签
            if isinstance(pred, (int, np.integer)):
                original_label = self.label_encoder.inverse_transform([pred])[0]
            else:
                original_label = pred

            result_item = {
                'id': item_id,
                'label': original_label
            }

            # 如果提供了文本，添加到结果中
            if texts is not None:
                result_item['text'] = texts[i]

            results.append(result_item)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        logger.info(f"预测结果已保存到: {output_file}")
        return results

    def load_predictions(self, prediction_file):
        """加载预测结果"""
        try:
            with open(prediction_file, 'r', encoding='utf-8') as f:
                predictions = json.load(f)
            logger.info(f"从 {prediction_file} 加载了 {len(predictions)} 条预测结果")
            return predictions
        except Exception as e:
            logger.error(f"加载预测结果失败: {e}")
            return []
