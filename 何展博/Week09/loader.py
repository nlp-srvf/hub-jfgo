# -*- coding: utf-8 -*-
import json
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""数据加载"""
class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        self.sentences = []  # 用于调试的原始句子列表
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                chars = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    chars.append(char)
                    labels.append(self.schema[label])
                
                self.sentences.append("".join(chars))
                self.process_sentence(chars, labels)
                
        print(f"成功加载 {len(self.data)} 个样本")
        print(f"示例样本: 原始句子: '{self.sentences[0]}', 处理后长度: {len(self.data[0][0])}")

    def process_sentence(self, chars, labels):
        """处理单个句子，确保标签与token对齐"""
        # 使用tokenizer处理字符列表（is_split_into_words=True表示输入是字符列表）
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.config["max_length"],
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        
        # 获取每个token对应的原始字符索引
        word_ids = encoding.word_ids(batch_index=0)
        
        # 为每个token创建对应的标签
        new_labels = []
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                # [CLS]、[SEP]等特殊token，标签设为-1
                new_labels.append(-1)
            else:
                # 保留原始标签（按字符顺序）
                new_labels.append(labels[word_id])
        
        # 验证长度是否一致
        assert len(new_labels) == len(encoding["input_ids"][0]), \
            f"标签长度 {len(new_labels)} 不等于 token长度 {len(encoding['input_ids'][0])}"
        
        # 保存处理后的数据
        self.data.append([
            encoding["input_ids"][0].squeeze(0),  # 转换为1D tensor
            torch.tensor(new_labels)
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

# 加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
    return token_dict

# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    # 示例配置（实际使用时请修改为你的配置）
    class Config:
        bert_path = "bert-base-chinese"
        schema_path = "schema.json"
        class_num = 10
        max_length = 128
        batch_size = 32
        use_crf = True
    
    # 测试数据加载
    dg = DataGenerator("../ner_data/train.txt", Config)
    dl = load_data("../ner_data/train.txt", Config)
    
    # 打印第一个batch的形状
    for batch in dl:
        input_ids, labels = batch
        print("输入形状:", input_ids.shape)
        print("标签形状:", labels.shape)
        print("示例标签:", labels[0][:10].tolist())
        break