# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import random
import jieba
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")   # \n\n是空行的换行符+上一行的换行符， unix下适用，window下是2个\r\n
            if self.config['model_type'].startswith('bert'):
                tokenizer = BertTokenizerFast.from_pretrained(self.config['bert_path'])
                for segment in segments:
                    sentence = []
                    labels = []
                    for line in segment.split('\n'):
                        if line.strip() == '':
                            continue
                        ch, label = line.split()
                        sentence.append(ch)
                        labels.append(self.schema[label])
                    self.sentences.append("".join(sentence))
                    sentence_encoding = tokenizer(sentence, is_split_into_words=True, truncation=True,
                                                  padding='max_length', max_length = self.config['max_length'])
                    tokens = sentence_encoding.tokens()
                    word_ids = sentence_encoding.word_ids()
                    word_ids = [-100 if wid is None else wid for wid in word_ids]
                    for index, token in enumerate(tokens):
                        if token in ['[CLS]', '[PAD]', '[SEP]']:
                            labels.insert(index, self.schema['BERT'])
                    labels = labels[:self.config['max_length']]
                    labels += [self.schema['BERT'] for _ in range(self.config['max_length'] - len(labels))]
                    # print('get tokens: ', tokens)
                    # print('get labels:', labels)
                    self.data.append([sentence_encoding['input_ids'], torch.LongTensor(labels), word_ids])
            else:
                for segment in segments:
                    sentence = []
                    labels = []
                    for line in segment.split("\n"):
                        if line.strip() == "":
                            continue
                        char, label = line.split()
                        sentence.append(char)
                        labels.append(self.schema[label])
                    self.sentences.append("".join(sentence))
                    input_ids = self.encode_sentence(sentence)
                    labels = self.padding(labels, -1) # 这里-1作为填充字符，而不是0，因为0是标签的一个，不能用于填充
                    self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

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

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return torch.LongTensor(self.data[index][0]), torch.LongTensor(self.data[index][1]), torch.LongTensor(self.data[index][2])

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
    dg = DataGenerator("./ner_data/train_copy", Config)
    dl = DataLoader(dg, batch_size=5, shuffle=True)
    for index, batched_data in enumerate(dl):
        input_ids, label, word_ids_ = batched_data
        print(input_ids.shape, ',', label.shape, ',', len(word_ids_))

