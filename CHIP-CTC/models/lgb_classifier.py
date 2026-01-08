import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import logging
import os
from typing import Dict, List, Tuple, Optional, Union

logger = logging.getLogger(__name__)

class LightGBMClassifier:
    """LightGBM分类器，用于CHIP-CTC任务"""

    def __init__(self, params=None):
        """
        初始化LightGBM分类器

        Args:
            params: LightGBM参数字典
        """
        self.default_params = {
            'objective': 'multiclass',
            'num_class': 44,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 100,
            'early_stopping_rounds': 20
        }

        if params:
            self.default_params.update(params)

        self.params = self.default_params.copy()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None

    def _prepare_data(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> Tuple[lgb.Dataset, Optional[np.ndarray]]:
        """
        准备LightGBM数据集

        Args:
            X: 特征矩阵
            y: 标签数组（可选）

        Returns:
            dataset: LightGBM数据集
            y_scaled: 归一化后的标签（如果是回归任务）
        """
        # 特征归一化
        if hasattr(self, 'scaler') and self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X) if y is not None else self.scaler.transform(X)
        else:
            X_scaled = X

        if y is not None:
            dataset = lgb.Dataset(X_scaled, label=y)
        else:
            dataset = lgb.Dataset(X_scaled)

        return dataset, None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: Optional[np.ndarray] = None, y_val: Optional[np.ndarray] = None,
            categorical_feature: Optional[List[str]] = None) -> 'LightGBMClassifier':
        """
        训练LightGBM模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征（可选）
            y_val: 验证标签（可选）
            categorical_feature: 类别特征列名列表

        Returns:
            self: 训练后的模型
        """
        logger.info("开始训练LightGBM模型...")

        # 准备训练数据
        train_dataset, _ = self._prepare_data(X_train, y_train)

        # 设置类别特征
        if categorical_feature:
            train_dataset.set_categorical_feature(categorical_feature)

        valid_sets = [train_dataset]
        valid_names = ['train']

        # 如果有验证集，添加到验证集中
        if X_val is not None and y_val is not None:
            val_dataset, _ = self._prepare_data(X_val, y_val)
            if categorical_feature:
                val_dataset.set_categorical_feature(categorical_feature)
            valid_sets.append(val_dataset)
            valid_names.append('valid')

        # 训练参数
        train_params = self.params.copy()

        # 如果有early_stopping_rounds，从参数中移除，在回调中使用
        early_stopping_rounds = train_params.pop('early_stopping_rounds', None)
        callbacks = []

        if early_stopping_rounds and len(valid_sets) > 1:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))
            callbacks.append(lgb.log_evaluation(10))

        # 训练模型
        self.model = lgb.train(
            train_params,
            train_dataset,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )

        logger.info("LightGBM模型训练完成")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别

        Args:
            X: 特征矩阵

        Returns:
            预测的类别标签
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 准备数据
        dataset, _ = self._prepare_data(X)
        predictions = self.model.predict(dataset.data)

        # 对于多分类，预测结果是概率数组，需要取最大概率的类别
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            predictions = np.argmax(predictions, axis=1)

        return predictions.astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率

        Args:
            X: 特征矩阵

        Returns:
            预测概率矩阵
        """
        if self.model is None:
            raise ValueError("模型未训练，请先调用fit方法")

        # 准备数据
        dataset, _ = self._prepare_data(X)
        probabilities = self.model.predict(dataset.data)

        # 如果是一维数组（二分类），转换为二维
        if len(probabilities.shape) == 1:
            probabilities = np.column_stack([1 - probabilities, probabilities])

        return probabilities

    def evaluate(self, X: np.ndarray, y: np.ndarray,
                target_names: Optional[List[str]] = None) -> Dict:
        """
        评估模型性能

        Args:
            X: 特征矩阵
            y: 真实标签
            target_names: 类别名称列表

        Returns:
            评估结果字典
        """
        predictions = self.predict(X)

        # 计算各种指标
        accuracy = accuracy_score(y, predictions)
        macro_f1 = f1_score(y, predictions, average='macro')
        weighted_f1 = f1_score(y, predictions, average='weighted')

        results = {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'predictions': predictions,
            'true_labels': y
        }

        logger.info(".4f")
        logger.info(".4f")
        logger.info(".4f")

        # 详细分类报告
        if target_names:
            report = classification_report(y, predictions, target_names=target_names)
        else:
            report = classification_report(y, predictions)

        logger.info("\n详细分类报告:")
        logger.info(report)
        results['classification_report'] = report

        return results

    def get_feature_importance(self, importance_type: str = 'split',
                             top_n: Optional[int] = None) -> pd.DataFrame:
        """
        获取特征重要性

        Args:
            importance_type: 重要性类型 ('split' 或 'gain')
            top_n: 返回前N个重要特征

        Returns:
            特征重要性DataFrame
        """
        if self.model is None:
            raise ValueError("模型未训练")

        importance = self.model.feature_importance(importance_type=importance_type)

        if self.feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(importance))]
        else:
            feature_names = self.feature_names

        # 创建DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        })

        # 排序
        importance_df = importance_df.sort_values('importance', ascending=False)

        if top_n:
            importance_df = importance_df.head(top_n)

        return importance_df

    def save_model(self, filepath: str):
        """
        保存模型

        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("没有可保存的模型")

        # 创建保存目录
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存模型和相关信息
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'params': self.params,
            'feature_names': self.feature_names
        }

        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str) -> 'LightGBMClassifier':
        """
        加载模型

        Args:
            filepath: 模型文件路径

        Returns:
            self: 加载后的模型
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data.get('scaler', StandardScaler())
        self.params = model_data.get('params', self.default_params)
        self.feature_names = model_data.get('feature_names')

        logger.info(f"模型已从 {filepath} 加载")
        return self

    def get_params(self) -> Dict:
        """获取模型参数"""
        return self.params.copy()

    def set_params(self, **params) -> 'LightGBMClassifier':
        """设置模型参数"""
        self.params.update(params)
        return self
