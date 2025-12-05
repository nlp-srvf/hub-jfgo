# -*- coding: utf-8 -*-
"""
RNN文本分类模型
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

from config import RNN_CONFIG, MODEL_DIR, MAX_SEQ_LENGTH
import os


class RNNClassifier(nn.Module):
    """RNN分类模型"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout, num_classes=2):
        super(RNNClassifier, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embedding_dim, 
            hidden_dim, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        output, hidden = self.rnn(embedded)  # output: (batch_size, seq_length, hidden_dim)
        
        # 取最后一个时间步的输出
        last_output = output[:, -1, :]  # (batch_size, hidden_dim)
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # (batch_size, num_classes)
        
        return logits


class RNNTrainer:
    def __init__(self, config=None):
        self.config = config or RNN_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = None
        self.learning_rate = self.config['learning_rate']
    
    def build_model(self, vocab_size=None):
        """构建模型"""
        vocab_size = vocab_size or self.config['vocab_size']
        self.model = RNNClassifier(
            vocab_size=vocab_size,
            embedding_dim=self.config['embedding_dim'],
            hidden_dim=self.config['hidden_dim'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        ).to(self.device)
        
        return self.model
    
    def train(self, train_loader, valid_loader, vocab_size):
        """训练模型"""
        print("\n" + "=" * 50)
        print("训练RNN模型...")
        print("=" * 50)
        
        self.build_model(vocab_size)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        
        best_valid_acc = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = train_correct / train_total
            
            # 验证阶段
            valid_acc = self._evaluate(valid_loader)
            
            print(f"Epoch {epoch+1}/{self.config['epochs']} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f} - "
                  f"Train Acc: {train_acc:.4f} - "
                  f"Valid Acc: {valid_acc:.4f}")
            
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                self.save_model()
        
        return best_valid_acc
    
    def _evaluate(self, data_loader):
        """评估模型"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        return correct / total
    
    def evaluate(self, test_loader):
        """评估测试集"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        acc = accuracy_score(all_labels, all_preds)
        print(f"\n测试集准确率: {acc:.4f}")
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, target_names=['差评', '好评']))
        
        return acc
    
    def predict_time(self, test_loader, n_samples=100):
        """测试预测100条数据的耗时"""
        self.model.eval()
        
        # 收集足够的样本
        all_inputs = []
        count = 0
        for batch in test_loader:
            all_inputs.append(batch['input_ids'])
            count += batch['input_ids'].size(0)
            if count >= n_samples:
                break
        
        inputs = torch.cat(all_inputs, dim=0)[:n_samples].to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(inputs)
        elapsed_time = time.time() - start_time
        
        return elapsed_time
    
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "rnn_model.pth")
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None, vocab_size=None):
        """加载模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "rnn_model.pth")
        self.build_model(vocab_size)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    from data_loader import (
        load_data, data_analysis, split_data, build_vocab, 
        get_dataloaders
    )
    from config import BATCH_SIZE
    
    # 加载和处理数据
    df = load_data()
    data_analysis(df)
    train_df, valid_df, test_df = split_data(df)
    vocab = build_vocab(train_df)
    
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, vocab, BATCH_SIZE
    )
    
    # 训练和评估
    trainer = RNNTrainer()
    trainer.train(train_loader, valid_loader, len(vocab))
    acc = trainer.evaluate(test_loader)
    pred_time = trainer.predict_time(test_loader)
    print(f"预测100条耗时: {pred_time:.4f}秒")

