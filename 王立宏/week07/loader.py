# coding: utf-8

'''
loader.py：加载数据集，做预处理，为训练做准备
'''

import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer=None, max_len=128, is_bert=False):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.is_bert = is_bert

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        if self.is_bert and self.tokenizer:
            encoding = self.tokenizer(
                self.texts[idx],
                truncation=True,
                padding='max_length',
                max_length=self.max_len,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
        else:
            return {
                'text': self.texts[idx],
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }

class DataLoaderProcessor:
    def __init__(self, config):
        self.config = config
        self.tokenizer = None
        if hasattr(config, 'bert_tokenizer_path'):
            self.tokenizer = BertTokenizer.from_pretrained(config.bert_tokenizer_path, local_files_only=True)

    def clean_text(self, text):
        """文本清洗"""
        text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        return text

    def cut_words(self, text):
        """分词"""
        # 使用jieba进行分词
        jieba.initialize()
        # 添加电商常用词到jieba词典（可选的优化）
        # 这些词在电商评论中常见，可以提高分词准确性
        custom_words = [
            '好评', '差评', '物流', '客服', '包装', '快递', '发货',
            '质量', '性价比', '服务态度', '配送', '售后', '五星'
        ]
        for word in custom_words:
            jieba.add_word(word)
        # 使用精确模式进行分词
        return list(jieba.cut(text, cut_all=False))

    def build_vocab(self, texts, min_freq=2):
        """构建词表"""
        word_freq = {}
        for text in texts:
            words = self.cut_words(text)
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

        vocab = {'<PAD>': 0, '<UNK>': 1}
        for word, freq in word_freq.items():
            if freq >= min_freq:
                vocab[word] = len(vocab)

        return vocab

    def text_to_sequence(self, text, vocab, max_len):
        """文本转序列"""
        words = self.cut_words(text)
        sequence = []
        for i, word in enumerate(words):
            if i < max_len:
                sequence.append(vocab.get(word, vocab['<UNK>']))

        if len(sequence) < max_len:
            sequence = sequence + [vocab['<PAD>']] * (max_len - len(sequence))

        return sequence[:max_len]

    def load_data(self):
        """加载数据并划分训练/测试集"""
        # 读取CSV文件
        df = pd.read_csv(self.config.data_path)

        # 清洗文本
        df['review'] = df['review'].apply(self.clean_text)

        # 划分训练和测试集
        texts = df['review'].values
        labels = df['label'].values

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=self.config.test_size,
            random_state=self.config.random_seed, stratify=labels
        )

        return X_train, X_test, y_train, y_test

    def load_all_data(self):
        """加载所有数据，不划分训练/测试集（用于多次重复评估）"""
        # 读取CSV文件
        df = pd.read_csv(self.config.data_path)

        # 清洗文本
        df['review'] = df['review'].apply(self.clean_text)

        # 返回完整数据集
        texts = df['review'].values
        labels = df['label'].values

        return texts, labels

    def get_bert_dataloader(self, texts, labels, shuffle=True):
        """获取BERT模型的数据加载器"""
        dataset = TextDataset(texts, labels, self.tokenizer, self.config.max_len, is_bert=True)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)

    def get_traditional_dataloader(self, texts, labels, vocab, shuffle=True):
        """获取传统模型的数据加载器"""
        # 文本转序列
        sequences = []
        for text in texts:
            seq = self.text_to_sequence(text, vocab, self.config.max_len)
            sequences.append(seq)

        # 转换为tensor
        sequences = torch.tensor(sequences, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)

        # 创建Dataset和DataLoader
        dataset = torch.utils.data.TensorDataset(sequences, labels)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=shuffle)