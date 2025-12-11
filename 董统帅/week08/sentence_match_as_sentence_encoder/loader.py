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
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.schema = load_schema(config["schema_path"])
        self.id_standard_question = {v: k for k,v in self.schema.items()}
        # print(self.id_standard_question)
        # self.train_data_size = config["epoch_data_size"] #由于采取随机采样，所以需要设定一个采样数量，否则可以一直采
        self.data_type = None  #用来标识加载的是训练集还是测试集 "train" or "test"
        self.load()

    def load(self):
        self.data = []
        self.knwb = defaultdict(list)
        self.standard_question_vec = {}
        self.knwb_center = defaultdict(list)
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
                        self.knwb_center[self.schema[label]].append(question)
                    standard_question_vector = self.encode_sentence(label)
                    self.standard_question_vec[self.schema[label]] = torch.LongTensor(standard_question_vector)
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

    #依照一定概率生成负样本或正样本
    #负样本从随机两个不同的标准问题中各随机选取一个
    #正样本从随机一个标准问题中随机选取两个
    def random_train_sample(self):
        standard_question_index = list(self.knwb.keys())
        # #随机正样本
        # if random.random() <= self.config["positive_sample_rate"]:
        #     p = random.choice(standard_question_index)
        #     #如果选取到的标准问下不足两个问题，则无法选取，所以重新随机一次
        #     if len(self.knwb[p]) < 2:
        #         return self.random_train_sample()
        #     else:
        #         s1, s2 = random.sample(self.knwb[p], 2)
        #         return [s1, s2, torch.LongTensor([1])]
        # #随机负样本
        # else:
        #     p, n = random.sample(standard_question_index, 2)
        #     s1 = random.choice(self.knwb[p])
        #     s2 = random.choice(self.knwb[n])
        #     return [s1, s2, torch.LongTensor([-1])]

        q1_index, q2_index = random.sample(standard_question_index, 2)
        q1_vec = self.standard_question_vec[q1_index]
        p_sample_index = random.choice(range(len(self.knwb[q1_index])))
        n_sample_index = random.choice(range(len(self.knwb[q2_index])))

        p_sample = self.knwb[q1_index][p_sample_index]
        n_sample = self.knwb[q2_index][n_sample_index]
        q1_standard= self.id_standard_question[q1_index]
        # print('generate sample a:', q1_standard, ' p:', self.knwb_center[q1_index][p_sample_index], ' n:', self.knwb_center[q2_index][n_sample_index])
        return q1_vec, p_sample, n_sample




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
    dg = DataGenerator("../data/train.json", Config)
    dl = DataLoader(dg, batch_size=Config["batch_size"], shuffle=True)
    for index, batch_data in enumerate(dl):
        a, p, n = batch_data
        print('a: ', a.squeeze())
        print('p: ', p.squeeze())
        print('n: ', n.squeeze())
    # dl = load_data(Config["valid_data_path"], Config, shuffle=False)
    # for sample in dl:
    #     print(sample)
