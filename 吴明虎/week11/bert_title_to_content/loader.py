# -*- coding: utf-8 -*-

import json
import re
import os
import torch
import numpy as np
from torch import dtype
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from transformers import BertTokenizer
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config, logger):
        self.config = config
        self.logger = logger
        self.path = data_path

        self.tokenizer = BertTokenizer.from_pretrained(self.config["bert_path"])
        self.vocab = self.tokenizer.vocab
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.max_length = self.config["input_max_length"]
        self.corpus = load_corpus(self.config["train_data_path"])
        self.load()


    def load(self):
        self.data = []
        for i, (prompt, answer) in enumerate(self.corpus):
            prompt_encode = self.tokenizer.encode(normalize_punctuation_and_whitespace(prompt), add_special_tokens=False)
            answer_encode = self.tokenizer.encode(normalize_punctuation_and_whitespace(answer), add_special_tokens=False)
            x = [self.tokenizer.cls_token_id] + prompt_encode + [self.tokenizer.sep_token_id] + answer_encode + [
                self.tokenizer.sep_token_id]
            y = len(prompt_encode) * [-1] + [-1] + answer_encode + [self.tokenizer.sep_token_id] + [-1]
            # 构建一个的mask矩阵，让prompt内可以交互，answer中上下文之间没有交互
            mask = self.create_mask(len(prompt_encode), len(answer_encode))
            # padding
            x = x[:self.max_length] + [0] * (self.max_length - len(x))
            y = y[:self.max_length] + [0] * (self.max_length - len(y))
            x = torch.LongTensor(x)
            y = torch.LongTensor(y)
            mask = self.pad_mask(mask, (self.max_length, self.max_length))

            # mask = self.get_mask(len(prompt_encode)+2,len(answer_encode)+1,self.max_length)
            self.data.append([x, mask, y])

    # 构造掩码，输入两个字符串的长度
    def create_mask(self,s1, s2):
        len_s1 = s1 + 2  # cls + sep
        len_s2 = s2 + 1  # sep
        # 创建掩码张量
        mask = torch.ones(len_s1 + len_s2, len_s1 + len_s2)
        # 遍历s1的每个token
        for i in range(len_s1):
            # s1的当前token不能看到s2的任何token
            mask[i, len_s1:] = 0
            # 遍历s2的每个token
        for i in range(len_s2):
            # s2的当前token不能看到后面的s2 token
            mask[len_s1 + i, len_s1 + i + 1:] = 0
        return mask

    def pad_mask(self,tensor, target_shape):
        # 获取输入张量和目标形状的长宽
        height, width = tensor.shape
        target_height, target_width = target_shape
        # 创建一个全零张量,形状为目标形状
        result = torch.zeros(target_shape, dtype=tensor.dtype, device=tensor.device)
        # 计算需要填充或截断的区域
        h_start = 0
        w_start = 0
        h_end = min(height, target_height)
        w_end = min(width, target_width)
        # 将原始张量对应的部分填充到全零张量中
        result[h_start:h_end, w_start:w_end] = tensor[:h_end - h_start, :w_end - w_start]
        return result

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [0] * (length - len(input_id))
        # input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    def get_mask(self, s1, s2, max_length):
        block1 = torch.ones(s1, s1, dtype=torch.int)
        block2 = torch.zeros(s1, s2, dtype=torch.int)
        block3 = torch.ones(s2, s1, dtype=torch.int)
        block4 = torch.tril(torch.ones(s2, s2, dtype=torch.int), diagonal=0)

        top_half = torch.cat([block1, block2], dim=1)
        bottom_half = torch.cat([block3, block4], dim=1)
        result_matrix = torch.cat([top_half, bottom_half], dim=0)

        current_size = result_matrix.shape[0]

        if current_size > max_length:
            result_matrix = result_matrix[:max_length, :max_length]
        elif current_size < max_length:
            padding_size = max_length - current_size
            # 删除未使用的 padding_matrix
            # 先进行水平填充
            result_matrix = torch.cat([result_matrix, torch.zeros(current_size, padding_size, dtype=torch.int)], dim=1)
            # 再进行垂直填充
            if padding_size > 0:
                bottom_padding = torch.zeros(padding_size, max_length, dtype=torch.int)
                result_matrix = torch.cat([result_matrix, bottom_padding], dim=0)

        return result_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    berttokenizer = BertTokenizer.from_pretrained(vocab_path)
    return berttokenizer.vocab

def load_corpus(path):
    corpus = []
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line:
                continue
            try:
                line = json.loads(line)
            except json.JSONDecodeError as e:
                continue
            corpus.append([line["title"], line["content"]])
    return corpus
def normalize_punctuation_and_whitespace(text):
    '''
    data clean
    '''
    # 1. 处理特殊符号
    text = text.replace('…', '...')  # 省略号
    text = text.replace('—', '--')  # 破折号（可选）
    text = text.replace('–', '-')  # en dash
    # 2. 全角标点转半角（包括全角空格）
    full_to_half = str.maketrans({
        '，': ',', '。': '.', '！': '!', '？': '?',
        '；': ';', '：': ':', '“': '"', '”': '"',
        '‘': "'", '’': "'", '（': '(', '）': ')',
        '【': '[', '】': ']', '《': '<', '》': '>',
        '　': ' ',  # 全角空格
    })
    text = text.translate(full_to_half)
    text = re.sub(r'\s+', '', text)  # 直接删除所有空格
    return text
#用torch自带的DataLoader类封装数据
def load_data(data_path, config, logger, shuffle=True):
    dg = DataGenerator(data_path, config, logger)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl



if __name__ == "__main__":
    from config import Config
    dl = load_data(Config["train_data_path"], Config, 1)
    print(dl[1])

