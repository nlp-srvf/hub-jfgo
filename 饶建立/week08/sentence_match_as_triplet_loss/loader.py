# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
"""
数据加载 - 三元组数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                #加载训练集
                if isinstance(line, dict):
                    self.data_type = "train"
                    questions = line["questions"]
                    label = line["target"]
                    for question in questions:
                        input_id = self.encode_sentence(question)
                        input_id = torch.LongTensor(input_id)
                        self.knwb[self.schema[label]].append(input_id)
                #加载测试集
                else:
                    self.data_type = "test"
                    assert isinstance(line, list)
                    question, label = line
                    input_id = self.encode_sentence(question)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([self.schema[label]])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        if self.config["vocab_path"] == "words.txt":
            for word in jieba.cut(text):
                input_id.append(self.vocab.get(word, self.vocab["[UNK]"]))
        else:
            for char in text:
                input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        if self.data_type == "train":
            return self.config["epoch_data_size"]
        else:
            assert self.data_type == "test", self.data_type
            return len(self.data)

    def __getitem__(self, index):
        if self.data_type == "train":
            return self.random_train_sample() #随机生成一个训练样本
        else:
            return self.data[index]

    # 随机生成一个三元组训练样本
    # anchor和positive从同一个标准问题中随机选取两个
    # negative从不同的标准问题中随机选取一个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        
        # 随机选择一个标准问题作为正样本类别
        p_index = random.choice(standard_question_index)
        
        # 如果选取到的标准问题下不足两个问题，则无法构成三元组，重新随机一次
        if len(self.knwb[p_index]) < 2:
            return self.random_train_sample()
        
        # 从正样本类别中随机选取两个问题作为anchor和positive
        anchor, positive = random.sample(self.knwb[p_index], 2)
        
        # 随机选择一个不同的标准问题作为负样本类别
        n_index = random.choice(standard_question_index)
        while n_index == p_index:  # 确保负样本类别与正样本类别不同
            n_index = random.choice(standard_question_index)
        
        # 从负样本类别中随机选取一个问题作为negative
        negative = random.choice(self.knwb[n_index])
        
        return [anchor, positive, negative]



#加载字表或词表
def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

#加载schema
def load_schema(schema_path):
    with open(schema_path, encoding="utf8") as f:
        return json.loads(f.read())

#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    # 测试数据加载
    Config["vocab_path"] = "../chars.txt"
    Config["train_data_path"] = "../data/train.json"
    Config["valid_data_path"] = "../data/valid.json"
    Config["schema_path"] = "../data/schema.json"
    
    dg = DataGenerator(Config["train_data_path"], Config)
    print("训练集样本数:", len(dg))
    print("知识库类别数:", len(dg.knwb))
    
    # 测试一个三元组样本
    anchor, positive, negative = dg[0]
    print("\nAnchor shape:", anchor.shape)
    print("Positive shape:", positive.shape)
    print("Negative shape:", negative.shape)
    
    # 测试DataLoader
    dl = load_data(Config["train_data_path"], Config)
    for batch in dl:
        anchor, positive, negative = batch
        print("\nBatch anchor shape:", anchor.shape)
        print("Batch positive shape:", positive.shape)
        print("Batch negative shape:", negative.shape)
        break

