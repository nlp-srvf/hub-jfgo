
import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.max_length = config['max_length']
        # self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer =  BertTokenizer.from_pretrained(config["bert_path"])
        self.sentences, self.labels = self.load()

    def load(self):
        # self.data = []
        sentences = []
        labels = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentence = []
                sentence_labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()[0], line.split()[1]
                    sentence.append(char)
                    sentence_labels.append(self.schema[label])
                sentences.append(sentence)
                labels.append(sentence_labels)
        return sentences, labels


                # 原代码
                # self.sentences.append("".join(sentenece))
                # input_ids = self.encode_sentence(sentenece)
                # labels = self.padding(labels, -1)
                # self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
    """
    # 不用考虑使用结巴分词
    def encode_sentence(self, text, padding=True):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        if padding:
            input_id = self.padding(input_id)
        return input_id
        """

# 补齐或截断输入的序列，使其可以在一个batch内运算
#     def padding(self, input_id, pad_token=0):
#         input_id = input_id[:self.config["max_length"]]
#         input_id += [pad_token] * (self.config["max_length"] - len(input_id))
#         return input_id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, index):
        chars = self.sentences[index]
        labels = self.labels[index]
        # === Step 1: Subword 分词并保持标签对齐 ===
        input_ids = []
        aligned_labels = []

        for char, label in zip(chars, labels):
            sub_tokens = self.tokenizer.tokenize(char)
            if not sub_tokens:  # 防止空 token
                sub_tokens = ["[UNK]"]

            input_ids.extend(self.tokenizer.convert_tokens_to_ids(sub_tokens))

            # 第一个 sub_token 保留原 label，其余设为 -1（训练时忽略）
            aligned_labels.append(label)
            aligned_labels.extend([-1] * (len(sub_tokens) - 1))

        # === Step 2: 添加 [CLS] 和 [SEP] ===
        input_ids = [self.tokenizer.cls_token_id] + input_ids + [self.tokenizer.sep_token_id]
        aligned_labels = [0] + aligned_labels + [0]  # CLS/SEP 对应的 label 可设为 O 类或忽略

        # === Step 3: Attention Mask ===
        attention_mask = [1] * len(input_ids)

        # === Step 4: Padding 到 max_length ===
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            aligned_labels = aligned_labels[:self.max_length]
        else:
            padding_length = self.max_length - len(input_ids)
            input_ids += [self.tokenizer.pad_token_id] * padding_length
            attention_mask += [0] * padding_length
            aligned_labels += [-1] * padding_length  # -1 表示 ignore
        # 断言检查
        assert len(input_ids) == self.max_length, f"input_ids 长度错误: {len(input_ids)}"
        assert len(attention_mask) == self.max_length, f"attention_mask 长度错误: {len(attention_mask)}"
        assert len(aligned_labels) == self.max_length, f"labels 长度错误: {len(aligned_labels)}"

        # === Step 5: 转为 tensor ===
        return {
            "input_ids": torch.LongTensor(input_ids),
            "attention_mask": torch.LongTensor(attention_mask),
            "labels": torch.LongTensor(aligned_labels)
        }


        # return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("ner_data/train", Config)
