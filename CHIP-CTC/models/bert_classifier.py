import torch
import torch.nn as nn
from transformers import (
    BertTokenizer, BertModel, BertConfig,
    RobertaTokenizer, RobertaModel, RobertaConfig,
    AutoTokenizer, AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report

logger = logging.getLogger(__name__)

class TextClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, texts, labels=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # 分词
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
        }

        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)

        return item

class BERTClassifier:
    """基于BERT的文本分类器"""

    def __init__(self, model_name='bert-base-chinese', num_labels=44, device=None, local_model_path=None):
        self.model_name = model_name
        self.num_labels = num_labels
        self.local_model_path = local_model_path
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        logger.info(f"使用设备: {self.device}")

        # 确定实际使用的模型路径
        actual_model_path = local_model_path if local_model_path else model_name
        logger.info(f"加载模型: {actual_model_path}")

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path)

        # 加载预训练模型
        self.model = AutoModelForSequenceClassification.from_pretrained(
            actual_model_path,
            num_labels=num_labels
        )

        self.model.to(self.device)

        # 设置为训练模式
        self.model.train()

    def create_data_loader(self, texts, labels=None, batch_size=16, shuffle=True):
        """创建数据加载器"""
        dataset = TextClassificationDataset(texts, labels, self.tokenizer)
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0  # Windows环境下设置为0
        )
        return data_loader

    def train(self, train_texts, train_labels, val_texts=None, val_labels=None,
              epochs=3, batch_size=16, learning_rate=2e-5, warmup_steps=0,
              class_weights=None, save_path=None):
        """训练模型"""

        logger.info("开始训练...")

        # 创建数据加载器
        train_loader = self.create_data_loader(
            train_texts, train_labels, batch_size=batch_size, shuffle=True
        )

        val_loader = None
        if val_texts is not None and val_labels is not None:
            val_loader = self.create_data_loader(
                val_texts, val_labels, batch_size=batch_size, shuffle=False
            )

        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        # 学习率调度器
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

        # 损失函数
        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()

        best_f1 = 0.0

        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")

            # 训练一个epoch
            train_loss = self._train_epoch(train_loader, optimizer, scheduler, criterion)

            logger.info(".4f")

            # 验证
            if val_loader is not None:
                val_metrics = self._evaluate(val_loader)
                logger.info(f"验证 Macro-F1: {val_metrics['macro_f1']:.4f}")

                # 保存最佳模型
                if val_metrics['macro_f1'] > best_f1:
                    best_f1 = val_metrics['macro_f1']
                    if save_path:
                        self.save_model(save_path)
                        logger.info(f"保存最佳模型到: {save_path}")

        logger.info("训练完成!")

    def _train_epoch(self, data_loader, optimizer, scheduler, criterion):
        """训练一个epoch"""
        self.model.train()
        total_loss = 0

        for batch in tqdm(data_loader, desc="Training"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)

            optimizer.zero_grad()

            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        return total_loss / len(data_loader)

    def _evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # 计算指标
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')
        weighted_f1 = f1_score(all_labels, all_predictions, average='weighted')

        return {
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'predictions': all_predictions,
            'labels': all_labels
        }

    def predict(self, texts, batch_size=16):
        """批量预测"""
        self.model.eval()

        data_loader = self.create_data_loader(
            texts, labels=None, batch_size=batch_size, shuffle=False
        )

        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)

                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        return np.array(all_predictions), np.array(all_probabilities)

    def predict_proba(self, texts, batch_size=16):
        """预测概率"""
        _, probabilities = self.predict(texts, batch_size=batch_size)
        return probabilities

    def save_model(self, save_path):
        """保存模型"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        logger.info(f"模型已保存到: {save_path}")

    def load_model(self, load_path):
        """加载模型"""
        self.model = AutoModelForSequenceClassification.from_pretrained(load_path)
        self.tokenizer = AutoTokenizer.from_pretrained(load_path)
        self.model.to(self.device)
        logger.info(f"模型已从 {load_path} 加载")

    def get_model_info(self):
        """获取模型信息"""
        return {
            'model_name': self.model_name,
            'num_labels': self.num_labels,
            'device': str(self.device),
            'model_parameters': sum(p.numel() for p in self.model.parameters())
        }
