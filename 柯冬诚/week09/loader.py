# -*- coding: utf-8 -*-

import json

import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer

"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.load()

    def load(self):
        self.data = []
        i = 0
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                sentenece = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    sentenece.append(char)
                    labels.append(self.schema[label])
                # 标签首尾添加[CLS]、[SEP]对应的 无关字符-8
                labels.insert(0, self.schema["O"])
                labels = self.padding(labels, -1)

                self.sentences.append("".join(sentenece))
                input_ids = self.encode_sentence(sentenece)
                # 句子首尾添加 [CLS]、[SEP]
                if len(sentenece) + 1 >= self.config["max_length"]:
                    input_ids[-1] = self.tokenizer.sep_token_id
                    labels[-1] = self.schema["O"]
                else:
                    input_ids[len(sentenece) + 1] = self.tokenizer.sep_token_id
                    labels[len(sentenece) + 1] = self.schema["O"]

                self.data.append([torch.LongTensor(input_ids), torch.LongTensor(labels)])
        return

    # 使用bert字典进行编码
    def encode_sentence(self, text, padding=True):
        input_id = []
        for char in text:
            input_id.append(self.tokenizer.convert_tokens_to_ids(char))
        input_id.insert(0, self.tokenizer.cls_token_id)
        if padding:
            input_id = self.padding(input_id)
        return input_id

    # 补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, pad_token=0):
        input_id = input_id[:self.config["max_length"]]
        input_id += [pad_token] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)


# 用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl


if __name__ == "__main__":
    from config import Config

    dg = DataGenerator("data/train.txt", Config)
