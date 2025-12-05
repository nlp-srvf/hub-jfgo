# -*- coding: utf-8 -*-
"""
数据加载和预处理模块
"""
import pandas as pd
import numpy as np
import json
import re
import jieba
from collections import Counter
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from config import (
    DATA_PATH, VOCAB_PATH, TRAIN_RATIO, VALID_RATIO, TEST_RATIO,
    RANDOM_SEED, MAX_VOCAB_SIZE, MAX_SEQ_LENGTH
)


def load_data():
    """加载原始数据"""
    df = pd.read_csv(DATA_PATH)
    return df


def clean_text(text):
    """文本清洗"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 去除多余空白字符
    text = re.sub(r'\s+', ' ', text)
    # 去除特殊字符（保留中文、英文、数字和常见标点）
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？、；：""''（）]', '', text)
    return text.strip()


def tokenize(text):
    """中文分词"""
    return list(jieba.cut(text))


def data_analysis(df):
    """数据分析"""
    print("=" * 50)
    print("数据分析报告")
    print("=" * 50)
    
    # 总样本数
    total_samples = len(df)
    print(f"\n总样本数: {total_samples}")
    
    # 正负样本数
    label_counts = df['label'].value_counts()
    positive_count = label_counts.get(1, 0)
    negative_count = label_counts.get(0, 0)
    print(f"好评数量: {positive_count} ({positive_count/total_samples*100:.2f}%)")
    print(f"差评数量: {negative_count} ({negative_count/total_samples*100:.2f}%)")
    print(f"正负样本比例: {positive_count/negative_count:.2f}:1" if negative_count > 0 else "")
    
    # 文本长度分析
    df['text_length'] = df['review'].apply(lambda x: len(str(x)) if pd.notna(x) else 0)
    avg_length = df['text_length'].mean()
    max_length = df['text_length'].max()
    min_length = df['text_length'].min()
    median_length = df['text_length'].median()
    
    print(f"\n文本长度统计:")
    print(f"  平均长度: {avg_length:.2f}")
    print(f"  最大长度: {max_length}")
    print(f"  最小长度: {min_length}")
    print(f"  中位数长度: {median_length}")
    
    # 按标签分析文本长度
    positive_avg_len = df[df['label'] == 1]['text_length'].mean()
    negative_avg_len = df[df['label'] == 0]['text_length'].mean()
    print(f"\n好评平均长度: {positive_avg_len:.2f}")
    print(f"差评平均长度: {negative_avg_len:.2f}")
    
    print("=" * 50)
    
    return {
        "total_samples": total_samples,
        "positive_count": positive_count,
        "negative_count": negative_count,
        "avg_length": avg_length,
        "max_length": max_length,
        "min_length": min_length
    }


def split_data(df):
    """划分训练集、验证集、测试集"""
    # 清洗文本
    df['clean_review'] = df['review'].apply(clean_text)
    
    # 分词
    df['tokens'] = df['clean_review'].apply(tokenize)
    
    # 先划分出测试集
    train_val_df, test_df = train_test_split(
        df, 
        test_size=TEST_RATIO, 
        random_state=RANDOM_SEED,
        stratify=df['label']
    )
    
    # 再从剩余数据中划分训练集和验证集
    relative_valid_ratio = VALID_RATIO / (TRAIN_RATIO + VALID_RATIO)
    train_df, valid_df = train_test_split(
        train_val_df,
        test_size=relative_valid_ratio,
        random_state=RANDOM_SEED,
        stratify=train_val_df['label']
    )
    
    print(f"\n数据集划分:")
    print(f"  训练集: {len(train_df)} 条")
    print(f"  验证集: {len(valid_df)} 条")
    print(f"  测试集: {len(test_df)} 条")
    
    return train_df, valid_df, test_df


def build_vocab(train_df, max_vocab_size=MAX_VOCAB_SIZE):
    """构建词表"""
    word_counts = Counter()
    for tokens in train_df['tokens']:
        word_counts.update(tokens)
    
    # 按频率排序，取前max_vocab_size个词
    most_common = word_counts.most_common(max_vocab_size - 2)  # 留两个位置给PAD和UNK
    
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for idx, (word, _) in enumerate(most_common, start=2):
        vocab[word] = idx
    
    # 保存词表
    with open(VOCAB_PATH, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    
    print(f"词表大小: {len(vocab)}")
    return vocab


def load_vocab():
    """加载词表"""
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocab = json.load(f)
    return vocab


def tokens_to_ids(tokens, vocab, max_length=MAX_SEQ_LENGTH):
    """将token转换为id序列"""
    ids = []
    for token in tokens[:max_length]:
        ids.append(vocab.get(token, vocab["<UNK>"]))
    
    # padding
    if len(ids) < max_length:
        ids.extend([vocab["<PAD>"]] * (max_length - len(ids)))
    
    return ids


class TextDataset(Dataset):
    """PyTorch数据集类"""
    def __init__(self, df, vocab, max_length=MAX_SEQ_LENGTH):
        self.labels = df['label'].values
        self.texts = df['tokens'].tolist()
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        tokens = self.texts[idx]
        ids = tokens_to_ids(tokens, self.vocab, self.max_length)
        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def get_dataloaders(train_df, valid_df, test_df, vocab, batch_size):
    """获取DataLoader"""
    train_dataset = TextDataset(train_df, vocab)
    valid_dataset = TextDataset(valid_df, vocab)
    test_dataset = TextDataset(test_df, vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, valid_loader, test_loader


def prepare_data_for_svm(train_df, valid_df, test_df):
    """为SVM准备数据（使用原始文本）"""
    X_train = train_df['clean_review'].values
    y_train = train_df['label'].values
    X_valid = valid_df['clean_review'].values
    y_valid = valid_df['label'].values
    X_test = test_df['clean_review'].values
    y_test = test_df['label'].values
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test


if __name__ == "__main__":
    # 测试数据加载
    df = load_data()
    stats = data_analysis(df)
    train_df, valid_df, test_df = split_data(df)
    vocab = build_vocab(train_df)

