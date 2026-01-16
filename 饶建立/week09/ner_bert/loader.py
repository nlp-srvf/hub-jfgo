# -*- coding: utf-8 -*-

import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载 - BERT版本
使用BERT的tokenizer进行数据处理
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # 使用BERT的tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        """
        加载NER数据
        数据格式：每行一个字符和标签，用空格分隔，句子之间用空行分隔
        """
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                if segment.strip() == "":
                    continue
                sentence = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    parts = line.split()
                    if len(parts) != 2:
                        continue
                    char, label = parts
                    sentence.append(char)
                    labels.append(self.schema[label])
                
                if len(sentence) == 0:
                    continue
                    
                # 保存原始句子（用于评估时解码）
                self.sentences.append("".join(sentence))
                
                # 使用BERT tokenizer编码
                # 注意：对于中文字符级别的NER，我们需要确保tokenizer不会把字符拆分
                input_ids, attention_mask, label_ids = self.encode_sentence(sentence, labels)
                
                self.data.append([
                    torch.LongTensor(input_ids),
                    torch.LongTensor(attention_mask),
                    torch.LongTensor(label_ids)
                ])
        return

    def encode_sentence(self, sentence_chars, labels):
        """
        使用BERT tokenizer对句子进行编码
        对于中文NER，需要特别处理以保持字符和标签的对齐
        
        :param sentence_chars: 字符列表
        :param labels: 标签列表
        :return: input_ids, attention_mask, label_ids
        """
        max_length = self.config["max_length"]
        
        # 添加[CLS]和[SEP]后的最大字符数
        max_char_length = max_length - 2  # 减去[CLS]和[SEP]
        
        # 截断句子和标签
        sentence_chars = sentence_chars[:max_char_length]
        labels = labels[:max_char_length]
        
        # 构建input_ids
        # [CLS] + 字符序列 + [SEP]
        tokens = ["[CLS]"] + sentence_chars + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 构建attention_mask
        attention_mask = [1] * len(input_ids)
        
        # 构建label_ids
        # [CLS]和[SEP]的标签设为-1（在计算损失时忽略）
        label_ids = [-1] + labels + [-1]
        
        # Padding
        padding_length = max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        label_ids += [-1] * padding_length  # padding位置的标签也设为-1
        
        return input_ids, attention_mask, label_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


def load_data(data_path, config, shuffle=True):
    """
    使用torch的DataLoader类封装数据
    """
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config
    
    print("Testing data loader...")
    dg = DataGenerator(Config["train_data_path"], Config)
    print(f"Dataset size: {len(dg)}")
    
    if len(dg) > 0:
        # 测试一个样本
        input_ids, attention_mask, labels = dg[0]
        print(f"\nSample 0:")
        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Original sentence: {dg.sentences[0][:50]}...")
        
        # 测试DataLoader
        dl = load_data(Config["train_data_path"], Config)
        for batch in dl:
            input_ids, attention_mask, labels = batch
            print(f"\nBatch shapes:")
            print(f"Input IDs: {input_ids.shape}")
            print(f"Attention mask: {attention_mask.shape}")
            print(f"Labels: {labels.shape}")
            break

