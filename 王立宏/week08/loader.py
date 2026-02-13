# coding: utf-8

'''
data_loader.py：加载数据集，做预处理，为训练做准备
'''

import json
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class Vocabulary:
    """词汇表类"""
    def __init__(self, vocab_path):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.unk_idx = 0
        self.pad_idx = 1
        
        # 加载词汇表
        self._load_vocab(vocab_path)
    
    def _load_vocab(self, vocab_path):
        """从文件加载词汇表"""
        with open(vocab_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 保留特殊标记的位置
        self.word_to_idx['[UNK]'] = self.unk_idx
        self.word_to_idx['[PAD]'] = self.pad_idx
        
        # 从第3行开始加载词汇（跳过[UNK]和[PAD]）
        idx = 2
        for line in lines[2:]:  # 跳过前两个特殊字符
            word = line.strip()
            if word:
                self.word_to_idx[word] = idx
                self.idx_to_word[idx] = word
                idx += 1
        
        # 添加特殊字符到反向映射
        self.idx_to_word[self.unk_idx] = '[UNK]'
        self.idx_to_word[self.pad_idx] = '[PAD]'
        
        print(f"词汇表大小: {len(self.word_to_idx)}")
    
    def text_to_indices(self, text, max_length=20):
        """将文本转换为索引序列"""
        if isinstance(text, str):
            # 按字符分割
            chars = list(text)
        else:
            chars = text
            
        indices = []
        for char in chars[:max_length]:
            idx = self.word_to_idx.get(char, self.unk_idx)
            indices.append(idx)
        
        # 填充或截断
        if len(indices) < max_length:
            indices.extend([self.pad_idx] * (max_length - len(indices)))
        
        return indices

class TextMatchingDataset(Dataset):
    """文本匹配数据集类，用于三元组损失"""
    def __init__(self, data_path, schema_path, vocab, max_length=20):
        self.vocab = vocab
        self.max_length = max_length
        self.triplets = []
        
        # 加载数据
        self._load_data(data_path, schema_path)
        self._generate_triplets()
        
        print(f"生成了 {len(self.triplets)} 个三元组样本")
    
    def _load_data(self, data_path, schema_path):
        """加载训练数据和标签映射"""
        # 加载标签映射
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        # 反转标签映射
        self.idx_to_label = {v: k for k, v in self.schema.items()}
        
        # 加载训练数据
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            if line.strip():  # 跳过空行
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict) and 'questions' in item:
                        # train.json格式
                        questions = item['questions']
                        target = item['target']
                        label = self.schema[target]
                        
                        for question in questions:
                            self.data.append({
                                'text': question,
                                'label': label,
                                'target': target
                            })
                    elif isinstance(item, list) and len(item) == 2:
                        # valid.json格式: [question, target]
                        question, target = item
                        label = self.schema[target]
                        
                        self.data.append({
                            'text': question,
                            'label': label,
                            'target': target
                        })
                except json.JSONDecodeError:
                    # 如果JSON解析失败，跳过该行
                    continue
        
        print(f"加载了 {len(self.data)} 条训练数据")
    
    def _generate_triplets(self):
        """生成三元组 (anchor, positive, negative)"""
        # 按类别组织数据
        label_to_texts = defaultdict(list)
        for item in self.data:
            label_to_texts[item['label']].append(item)
        
        # 为每个样本生成三元组
        for item in self.data:
            anchor_text = item['text']
            anchor_label = item['label']
            
            # 选择正样本：同类别的其他样本
            positive_candidates = [t for t in label_to_texts[anchor_label] 
                                 if t['text'] != anchor_text]
            if positive_candidates:
                positive = random.choice(positive_candidates)
            else:
                # 如果没有其他正样本，自己作为正样本
                positive = item
            
            # 选择负样本：不同类别的样本
            negative_labels = [l for l in label_to_texts.keys() if l != anchor_label]
            if negative_labels:
                negative_label = random.choice(negative_labels)
                negative = random.choice(label_to_texts[negative_label])
            else:
                # 如果没有负样本（理论上不应该发生）
                negative = item
            
            self.triplets.append({
                'anchor': anchor_text,
                'positive': positive['text'],
                'negative': negative['text'],
                'anchor_label': anchor_label
            })
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        
        # 转换文本为索引序列
        anchor_indices = self.vocab.text_to_indices(triplet['anchor'], self.max_length)
        positive_indices = self.vocab.text_to_indices(triplet['positive'], self.max_length)
        negative_indices = self.vocab.text_to_indices(triplet['negative'], self.max_length)
        
        return {
            'anchor': torch.tensor(anchor_indices, dtype=torch.long),
            'positive': torch.tensor(positive_indices, dtype=torch.long),
            'negative': torch.tensor(negative_indices, dtype=torch.long),
            'anchor_label': torch.tensor(triplet['anchor_label'], dtype=torch.long)
        }

class TextMatchingEvalDataset(Dataset):
    """用于评估的数据集类"""
    def __init__(self, data_path, schema_path, vocab, max_length=20):
        self.vocab = vocab
        self.max_length = max_length
        
        # 加载数据
        with open(schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
        
        with open(data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        self.data = []
        for line in lines:
            if line.strip():  # 跳过空行
                try:
                    item = json.loads(line.strip())
                    if isinstance(item, dict) and 'questions' in item:
                        # train.json格式
                        questions = item['questions']
                        target = item['target']
                        label = self.schema[target]
                        
                        for question in questions:
                            self.data.append({
                                'text': question,
                                'label': label,
                                'target': target
                            })
                    elif isinstance(item, list) and len(item) == 2:
                        # valid.json格式: [question, target]
                        question, target = item
                        label = self.schema[target]
                        
                        self.data.append({
                            'text': question,
                            'label': label,
                            'target': target
                        })
                except json.JSONDecodeError:
                    # 如果JSON解析失败，跳过该行
                    continue
        
        print(f"加载了 {len(self.data)} 条评估数据")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # 转换文本为索引序列
        indices = self.vocab.text_to_indices(item['text'], self.max_length)
        
        return {
            'text': torch.tensor(indices, dtype=torch.long),
            'label': torch.tensor(item['label'], dtype=torch.long),
            'target': item['target']
        }

def collate_fn(batch):
    """自定义批次处理函数"""
    anchors = torch.stack([item['anchor'] for item in batch])
    positives = torch.stack([item['positive'] for item in batch])
    negatives = torch.stack([item['negative'] for item in batch])
    anchor_labels = torch.stack([item['anchor_label'] for item in batch])
    
    return {
        'anchors': anchors,
        'positives': positives,
        'negatives': negatives,
        'labels': anchor_labels
    }

def eval_collate_fn(batch):
    """评估数据的批次处理函数"""
    texts = torch.stack([item['text'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    targets = [item['target'] for item in batch]
    
    return {
        'texts': texts,
        'labels': labels,
        'targets': targets
    }

def get_data_loaders(config):
    """获取数据加载器"""
    # 初始化词汇表
    vocab = Vocabulary(config['vocab_path'])
    
    # 创建数据集
    train_dataset = TextMatchingDataset(
        config['train_data_path'],
        config['schema_path'],
        vocab,
        config['max_length']
    )
    
    valid_dataset = TextMatchingEvalDataset(
        config['valid_data_path'],
        config['schema_path'],
        vocab,
        config['max_length']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=eval_collate_fn,
        num_workers=0
    )
    
    return train_loader, valid_loader, vocab

if __name__ == "__main__":
    # 测试代码
    from config import Config
    
    train_loader, valid_loader, vocab = get_data_loaders(Config)
    
    print(f"训练批次数: {len(train_loader)}")
    print(f"验证批次数: {len(valid_loader)}")
    
    # 查看一个批次的数据
    for batch in train_loader:
        print(f"Anchor shape: {batch['anchors'].shape}")
        print(f"Positive shape: {batch['positives'].shape}")
        print(f"Negative shape: {batch['negatives'].shape}")
        print(f"Labels shape: {batch['labels'].shape}")
        break

