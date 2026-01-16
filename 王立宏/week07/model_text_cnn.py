# coding: utf-8

'''
TextCNN：定义神经网络模型结构
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_filters, filter_sizes, num_classes, dropout=0.5):
        super(TextCNN, self).__init__()
        # 词嵌入
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 卷积层
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embedding_dim)) for k in filter_sizes
        ])
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
        # 卷积操作
        conv_result = []
        for conv in self.convs:
            conv_out = torch.relu(conv(embedded))  # (batch, num_filters, new_seq_len, 1)
            conv_out = conv_out.squeeze(3)  # (batch, num_filters, new_seq_len)
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2))  # (batch, num_filters, 1)
            conv_result.append(pool_out.squeeze(2))  # (batch, num_filters)
        # 拼接所有卷积结果
        output = torch.cat(conv_result, dim=1)  # (batch, num_filters * len(filter_sizes))
        output = self.dropout(output)
        output = self.fc(output)  # (batch, num_classes)
        return output

class TextCNNClassifier:
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.device = config.device
        # 初始化模型
        self.model = TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config.textcnn_embedding_dim,
            num_filters=config.textcnn_num_filters,
            filter_sizes=config.textcnn_filter_sizes,
            num_classes=2,
            dropout=config.textcnn_dropout
        )
        self.model.to(self.device)
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.textcnn_lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader):
        """训练模型"""
        start_time = time.time()
        train_start_time = time.time()
        for epoch in range(self.config.textcnn_epochs):
            self.model.train()
            total_loss = 0
            for batch_idx, (data, labels) in enumerate(train_dataloader):
                # 将数据移到GPU
                data = data.to(self.device)
                labels = labels.to(self.device)
                # 清零梯度
                self.optimizer.zero_grad()
                # 前向传播
                outputs = self.model(data)
                loss = self.criterion(outputs, labels)
                # 反向传播
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_dataloader)
            print(f'TextCNN Epoch {epoch+1}/{self.config.textcnn_epochs}, Loss: {avg_loss:.4f}')
        training_time = time.time() - train_start_time
        print(f'TextCNN training completed in {training_time:.2f} seconds')
        self.training_time = time.time() - start_time
        return training_time

    def predict(self, dataloader):
        """预测"""
        self.model.eval()
        predictions = []
        probabilities = []
        inference_start_time = time.time()
        with torch.no_grad():
            for data, _ in dataloader:
                data = data.to(self.device)
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        inference_time = time.time() - inference_start_time
        return np.array(predictions), np.array(probabilities), inference_time

    def evaluate(self, dataloader):
        """评估模型"""
        predictions, probabilities, inference_time = self.predict(dataloader)
        # 获取真实标签
        true_labels = []
        for _, labels in dataloader:
            true_labels.extend(labels.numpy())
        # 计算指标
        accuracy = accuracy_score(true_labels, predictions)
        report = classification_report(true_labels, predictions, output_dict=True)
        return accuracy, report, inference_time

    def save_model(self, path):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
        print(f'TextCNN model saved to {path}')

    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'TextCNN model loaded from {path}')