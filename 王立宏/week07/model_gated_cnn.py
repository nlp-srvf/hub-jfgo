# coding: utf-8

'''
门控卷积神经网络：定义神经网络模型结构
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report

class GatedCNN(nn.Module):
    """门控CNN模型"""
    def __init__(self, vocab_size, embedding_dim, num_filters, kernel_size, output_dim, dropout=0.1):
        super(GatedCNN, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # 主卷积层
        self.conv = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(kernel_size, embedding_dim),
            padding=(kernel_size//2, 0)
        )
        # 门控卷积层
        self.conv_gate = nn.Conv2d(
            in_channels=1,
            out_channels=num_filters,
            kernel_size=(kernel_size, embedding_dim),
            padding=(kernel_size//2, 0)
        )
        # 全连接层
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters, output_dim)

    def forward(self, x):
        # x: (batch, seq_len)
        # 词嵌入
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.unsqueeze(1)  # (batch, 1, seq_len, embedding_dim)
        # 主卷积
        A = self.conv(embedded)  # (batch, num_filters, seq_len, 1)
        A = A.squeeze(3)  # (batch, num_filters, seq_len)
        # 门控卷积
        B = self.conv_gate(embedded)  # (batch, num_filters, seq_len, 1)
        B = B.squeeze(3)  # (batch, num_filters, seq_len)
        B = torch.sigmoid(B)  # 门控信号
        # 门控机制
        output = A * B  # 逐元素相乘
        output = torch.tanh(output)
        # 全局最大池化
        output = torch.max_pool1d(output, output.size(2))  # (batch, num_filters, 1)
        output = output.squeeze(2)  # (batch, num_filters)
        # Dropout
        output = self.dropout(output)
        # 全连接层
        output = self.fc(output)  # (batch, num_classes)
        return output

class GatedCNNClassifier:
    def __init__(self, config, vocab_size):
        self.config = config
        self.vocab_size = vocab_size
        self.device = config.device
        # 初始化模型
        self.model = GatedCNN(
            vocab_size=vocab_size,
            embedding_dim=config.gatedcnn_embedding_dim,
            num_filters=config.gatedcnn_num_filters,
            kernel_size=config.gatedcnn_kernel_size,
            output_dim=2
        )
        self.model.to(self.device)
        # 优化器和损失函数
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.gatedcnn_lr)
        self.criterion = nn.CrossEntropyLoss()

    def train(self, train_dataloader):
        """训练模型"""
        start_time = time.time()
        train_start_time = time.time()
        for epoch in range(self.config.gatedcnn_epochs):
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
            print(f'GatedCNN Epoch {epoch+1}/{self.config.gatedcnn_epochs}, Loss: {avg_loss:.4f}')
        training_time = time.time() - train_start_time
        print(f'GatedCNN training completed in {training_time:.2f} seconds')
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
        print(f'GatedCNN model saved to {path}')

    def load_model(self, path):
        """加载模型"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f'GatedCNN model loaded from {path}')