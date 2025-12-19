# coding: utf-8

'''
loader.py：加载数据集，做预处理，为训练做准备
'''

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast
import json
from typing import List, Tuple, Dict, Any
import numpy as np


class NERDataset(Dataset):
    """
    NER数据集类
    用于处理命名实体识别任务的数据加载和预处理
    """
    
    def __init__(self, 
                 data_path: str, 
                 tokenizer: BertTokenizerFast, 
                 label2id: Dict[str, int],
                 max_length: int = 128):
        """
        初始化NER数据集
        
        Args:
            data_path: 数据文件路径
            tokenizer: BERT分词器
            label2id: 标签到ID的映射字典
            max_length: 最大序列长度
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self.samples = self._load_data()
    
    def _load_data(self) -> List[Dict[str, Any]]:
        """
        加载NER数据
        数据格式：每行包含一个字符和对应的标签，用空格分隔
        句子之间用空行分隔
        
        Returns:
            处理后的样本列表
        """
        samples = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_tokens = []
        current_labels = []
        
        for line in lines:
            line = line.strip()
            if not line:  # 空行表示句子结束
                if current_tokens:  # 保存当前句子
                    samples.append({
                        'tokens': current_tokens,
                        'labels': current_labels
                    })
                    current_tokens = []
                    current_labels = []
            else:
                try:
                    token, label = line.split()
                    current_tokens.append(token)
                    current_labels.append(label)
                except ValueError:
                    # 如果分割失败，跳过该行
                    continue
        
        # 处理最后一个句子
        if current_tokens:
            samples.append({
                'tokens': current_tokens,
                'labels': current_labels
            })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            包含模型输入的字典
        """
        sample = self.samples[idx]
        tokens = sample['tokens']
        labels = sample['labels']
        
        # 使用BERT分词器进行编码
        # 注意：对于中文，我们直接使用字符级别的分词
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理标签对齐
        word_ids = encoded.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], [PAD]等特殊token
                aligned_labels.append(-100)  # -100会被CrossEntropyLoss忽略
            elif word_idx != previous_word_idx:
                # 新词的开始
                if word_idx < len(labels):
                    aligned_labels.append(self.label2id[labels[word_idx]])
                else:
                    aligned_labels.append(-100)
            else:
                # 同一个词的后续部分（对于中文基本不会出现，因为中文字符不会被拆分）
                aligned_labels.append(-100)
            
            previous_word_idx = word_idx
        
        # 返回模型输入
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'token_type_ids': encoded['token_type_ids'].squeeze(0),
            'labels': torch.tensor(aligned_labels, dtype=torch.long),
            'tokens': tokens,  # 保存原始tokens用于评估
            'original_labels': labels  # 保存原始labels用于评估
        }


class NERDataLoader:
    """
    NER数据加载器类
    负责创建训练、验证和测试的数据加载器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化数据加载器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.tokenizer = BertTokenizerFast.from_pretrained(config['bert_path'])
        self.label2id, self.id2label = self._load_schema()
        
    def _load_schema(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        加载标签schema
        
        Returns:
            label2id: 标签到ID的映射
            id2label: ID到标签的映射
        """
        with open(self.config['schema_path'], 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        label2id = schema
        id2label = {v: k for k, v in schema.items()}
        
        return label2id, id2label
    
    def get_datasets(self) -> Tuple[NERDataset, NERDataset]:
        """
        获取训练和验证数据集
        
        Returns:
            train_dataset, valid_dataset
        """
        train_dataset = NERDataset(
            data_path=self.config['train_data_path'],
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.config['max_length']
        )
        
        valid_dataset = NERDataset(
            data_path=self.config['valid_data_path'],
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            max_length=self.config['max_length']
        )
        
        return train_dataset, valid_dataset
    
    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练和验证数据加载器
        
        Returns:
            train_dataloader, valid_dataloader
        """
        train_dataset, valid_dataset = self.get_datasets()
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        valid_dataloader = DataLoader(
            valid_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        return train_dataloader, valid_dataloader
    
    def _collate_fn(self, batch):
        """
        自定义批处理函数
        
        Args:
            batch: 批量样本
            
        Returns:
            处理后的批量数据
        """
        # 将batch中的数据堆叠
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': labels,
            'tokens': [item['tokens'] for item in batch],
            'original_labels': [item['original_labels'] for item in batch]
        }
    
    def get_label_mappings(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        获取标签映射字典
        
        Returns:
            label2id, id2label
        """
        return self.label2id, self.id2label


def test_data_loader():
    """
    测试数据加载器功能
    """
    from config import Config
    
    print("🧪 测试数据加载器...")
    
    # 创建数据加载器
    data_loader = NERDataLoader(Config)
    
    # 获取标签映射
    label2id, id2label = data_loader.get_label_mappings()
    print(f"📊 标签数量: {len(label2id)}")
    print(f"🏷️  标签映射: {label2id}")
    
    # 获取数据加载器
    train_dataloader, valid_dataloader = data_loader.get_dataloaders()
    
    # 测试训练数据
    print(f"📈 训练批次数: {len(train_dataloader)}")
    print(f"📊 验证批次数: {len(valid_dataloader)}")
    
    # 获取一个batch测试
    for batch in train_dataloader:
        print(f"🔍 Input IDs形状: {batch['input_ids'].shape}")
        print(f"🎯 Attention Mask形状: {batch['attention_mask'].shape}")
        print(f"📋 Labels形状: {batch['labels'].shape}")
        print(f"📝 第一个样本tokens: {batch['tokens'][0]}")
        print(f"🏷️  第一个样本labels: {batch['original_labels'][0]}")
        break
    
    print("✅ 数据加载器测试完成！")


if __name__ == "__main__":
    test_data_loader()
