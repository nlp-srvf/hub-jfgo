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

        self.vocab = load_vocab(config["bert_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.config["pad_idx"] = self.vocab["[PAD]"]
        self.config["start_idx"] = self.vocab["[CLS]"]
        self.config["end_idx"] = self.vocab["[SEP]"]
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for i, line in enumerate(f):
                if not line:
                    continue
                try:
                    line = json.loads(line)
                except json.JSONDecodeError as e:
                    continue
                title = normalize_punctuation_and_whitespace(line["title"])
                content = normalize_punctuation_and_whitespace(line["content"])
                self.prepare_data(title, content)
        return


    #输入输出转化成序列
    def prepare_data(self, title, content):
        input_seq,predict_seq,mask = self.encode_sentence(title, content, self.config["input_max_length"],) #输入序列
        self.data.append([torch.LongTensor(input_seq),
                          torch.LongTensor(predict_seq),
                          torch.Tensor(mask)])

    #     return
    # def load(self):
    #     print("0000")
    #     self.data = []
    #     with open(self.path, encoding="utf8") as f:
    #         lines = f.readlines()
    #         print("0000")
    #         self.logger.info("加载数据集...---",len(lines))
    #         for line_num, line in enumerate(lines, 1):
    #             line = line.strip()
    #             if not line:  # 跳过空行
    #                 continue
    #             if line.startswith('\ufeff'):  # 移除BOM
    #                 line = line[1:]
    #
    #             try:
    #                 line_data = json.loads(line)
    #                 yield data
    #             except json.JSONDecodeError as e:
    #                 print(f"第{line_num}行JSON解析失败: {line}")
    #                 continue
    #             title = line_data["title"]
    #             content = line_data["content"]
    #             input_seq, predict_seq,mask = self.encode_sentence(title, content, self.config["input_max_length"])  # 输入序列
    #             self.data.append([torch.LongTensor(input_seq),
    #                               torch.LongTensor(predict_seq),
    #                               torch.tensor(mask, dtype=torch.long)
    #                               ])
    #     return

    #文本到对应的index
    #头尾分别加入[cls]和[sep]
    def encode_sentence(self,title , content, max_length, ):
        input_id = []
        title_len = len(title)
        content_len = len(content)
        for char in title:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id.append(self.vocab["[SEP]"])
        for char in content:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id, max_length)
        predict_seq = [self.vocab["[UNK]"]] * title_len  # 添加标题长度的 UNK 序列
        predict_seq.extend([self.vocab.get(char, self.vocab["[UNK]"]) for char in content])
        # predict_seq = [self.vocab.get(char, self.vocab["[UNK]"]) for char in content]
        # # predict_seq.append(self.vocab["[SEP]"])
        predict_seq = self.padding(predict_seq, max_length)

        att_mask = self.get_mask(title_len, content_len+1,max_length)

        return input_id, predict_seq,att_mask

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id, length):
        input_id = input_id[:length]
        input_id += [self.vocab["[PAD]"]] * (length - len(input_id))
        return input_id

    def get_mask(self,s1,s2,max_length):
        block1 = torch.ones(s1, s1, dtype=torch.int)

        # 第二块: s1 x s2 全为0
        block2 = torch.zeros(s1, s2, dtype=torch.int)

        # 第三块: s2 x s1 全为1
        block3 = torch.ones(s2, s1, dtype=torch.int)

        # 第四块: s2 x s2 下三角矩阵
        block4 = torch.tril(torch.ones(s2, s2, dtype=torch.int), diagonal=0)  # 下三角为1，上三角为0

        # 水平拼接上半部分 [block1 | block2]
        top_half = torch.cat([block1, block2], dim=1)

        # 水平拼接下半部分 [block3 | block4]
        bottom_half = torch.cat([block3, block4], dim=1)

        # 垂直拼接整体矩阵 [top_half; bottom_half]
        result_matrix = torch.cat([top_half, bottom_half], dim=0)
        # 扩展到max_length，其余部分填充为0

        current_size = result_matrix.size(0)
        if current_size < max_length:
            # 创建一个max_length x max_length的全零矩阵
            expanded_matrix = torch.zeros(max_length, max_length, dtype=torch.int)
            # 将原矩阵复制到新矩阵的左上角
            expanded_matrix[:current_size, :current_size] = result_matrix
            result_matrix = expanded_matrix
        elif current_size > max_length:
            # 如果当前矩阵比max_length大，则截断
            result_matrix = result_matrix[:max_length, :max_length]

        return result_matrix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]


def load_vocab(vocab_path):
    berttokenizer = BertTokenizer.from_pretrained(vocab_path)
    return berttokenizer.vocab
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

