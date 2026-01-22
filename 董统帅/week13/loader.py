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
        self.tokenizer = BertTokenizer.from_pretrained(self.config['pretrain_model_path'])
        self.schema = self.load_schema(config["schema_path"])
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")   # \n\n是空行的换行符+上一行的换行符， unix下适用，window下是2个\r\n
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
                input_ids = self.encode_sentence(sentence)
                labels = self.padding(labels, pad_token=-100)
                # print(sentence, input_ids, labels)
                self.data.append((torch.LongTensor(input_ids), torch.LongTensor(labels)))

        return

    def encode_sentence(self, text, padding=True):
        input_id = [self.tokenizer.vocab.get(key, self.tokenizer.vocab.get('[UNK]')) for key in text]
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
        return torch.LongTensor(self.data[index][0]), torch.LongTensor(self.data[index][1])

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
    dl = DataLoader(dg, batch_size=2, shuffle=True)
    for index, batched_data in enumerate(dl):
        input_ids, label = batched_data
        # print(input_ids, label)
        print(input_ids.shape, ',', label.shape)