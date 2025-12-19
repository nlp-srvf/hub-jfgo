# coding: utf-8

'''
BERT：定义神经网络模型结构
'''

import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from sklearn.metrics import accuracy_score, classification_report

class BertClassifier:
    def __init__(self, config, num_labels=2):
        self.config = config
        self.num_labels = num_labels
        self.device = config.device

        # 加载本地预训练的BERT模型
        self.model = BertForSequenceClassification.from_pretrained(
            config.bert_local_path,
            num_labels=num_labels,
            local_files_only=True
        )
        self.model.to(self.device)

    def train(self, train_dataloader, val_dataloader=None):
        start_time = time.time()

        # 优化器和学习率调度器
        optimizer = AdamW(self.model.parameters(), lr=self.config.bert_lr)
        total_steps = len(train_dataloader) * self.config.bert_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        self.model.train()
        train_start_time = time.time()

        for epoch in range(self.config.bert_epochs):
            total_loss = 0
            for batch in train_dataloader:
                # 将数据移到GPU
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # 清零梯度
                self.model.zero_grad()

                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                loss = outputs.loss
                logits = outputs.logits

                # 反向传播
                loss.backward()

                # 更新参数
                optimizer.step()
                scheduler.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_dataloader)
            print(f'Epoch {epoch+1}/{self.config.bert_epochs}, Loss: {avg_loss:.4f}')

        training_time = time.time() - train_start_time
        print(f'BERT training completed in {training_time:.2f} seconds')

        self.training_time = time.time() - start_time
        return training_time

    def predict(self, dataloader):
        """预测"""
        self.model.eval()
        predictions = []
        probabilities = []

        inference_start_time = time.time()

        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())

        inference_time = time.time() - inference_start_time

        return np.array(predictions), np.array(probabilities), inference_time

    def evaluate(self, dataloader):
        """评估模型"""
        predictions, probabilities, inference_time = self.predict(dataloader)

        # 获取真实标签
        true_labels = []
        for batch in dataloader:
            true_labels.extend(batch['labels'].numpy())

        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)

        return accuracy, report, inference_time

    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f'Model saved to {path}')

    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'Model loaded from {path}')