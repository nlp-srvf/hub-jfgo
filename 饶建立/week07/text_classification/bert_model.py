# -*- coding: utf-8 -*-
"""
BERT文本分类模型
"""
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from transformers import BertTokenizer, BertModel, BertForSequenceClassification

from config import BERT_CONFIG, MODEL_DIR
import os


class BertDataset(Dataset):
    """BERT数据集类"""
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BertClassifier(nn.Module):
    """BERT分类模型"""
    def __init__(self, model_name, num_classes=2):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token的输出
        pooled_output = self.dropout(pooled_output)
        logits = self.fc(pooled_output)
        return logits


class BertTrainer:
    def __init__(self, config=None):
        self.config = config or BERT_CONFIG
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model = None
        self.tokenizer = None
        self.learning_rate = self.config['learning_rate']
    
    def build_model(self):
        """构建模型"""
        print("加载BERT预训练模型...")
        self.tokenizer = BertTokenizer.from_pretrained(self.config['model_name'])
        self.model = BertClassifier(self.config['model_name']).to(self.device)
        return self.model
    
    def prepare_data(self, train_df, valid_df, test_df):
        """准备数据"""
        train_dataset = BertDataset(
            train_df['clean_review'].values,
            train_df['label'].values,
            self.tokenizer,
            self.config['max_length']
        )
        valid_dataset = BertDataset(
            valid_df['clean_review'].values,
            valid_df['label'].values,
            self.tokenizer,
            self.config['max_length']
        )
        test_dataset = BertDataset(
            test_df['clean_review'].values,
            test_df['label'].values,
            self.tokenizer,
            self.config['max_length']
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=self.config['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=False)
        
        return train_loader, valid_loader, test_loader
    
    def train(self, train_loader, valid_loader):
        """训练模型"""
        print("\n" + "=" * 50)
        print("训练BERT模型...")
        print("=" * 50)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        
        best_valid_acc = 0
        
        for epoch in range(self.config['epochs']):
            # 训练阶段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                if (batch_idx + 1) % 50 == 0:
                    print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")
            
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
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
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
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
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
        all_input_ids = []
        all_attention_masks = []
        count = 0
        
        for batch in test_loader:
            all_input_ids.append(batch['input_ids'])
            all_attention_masks.append(batch['attention_mask'])
            count += batch['input_ids'].size(0)
            if count >= n_samples:
                break
        
        input_ids = torch.cat(all_input_ids, dim=0)[:n_samples].to(self.device)
        attention_mask = torch.cat(all_attention_masks, dim=0)[:n_samples].to(self.device)
        
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(input_ids, attention_mask)
        elapsed_time = time.time() - start_time
        
        return elapsed_time
    
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "bert_model.pth")
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "bert_model.pth")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    from data_loader import load_data, data_analysis, split_data
    
    # 加载和处理数据
    df = load_data()
    data_analysis(df)
    train_df, valid_df, test_df = split_data(df)
    
    # 训练和评估
    trainer = BertTrainer()
    trainer.build_model()
    train_loader, valid_loader, test_loader = trainer.prepare_data(train_df, valid_df, test_df)
    trainer.train(train_loader, valid_loader)
    acc = trainer.evaluate(test_loader)
    pred_time = trainer.predict_time(test_loader)
    print(f"预测100条耗时: {pred_time:.4f}秒")

