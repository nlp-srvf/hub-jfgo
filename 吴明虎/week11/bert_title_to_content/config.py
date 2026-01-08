# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "model_path": "output",
    "input_max_length": 100,
    "output_max_length": 100,
    "epoch": 200,
    "batch_size": 32,
    "optimizer": "adam",
    "learning_rate": 5e-5,
    "seed":42,
    "vocab_size":6219,
    "vocab_path":"vocab.txt",
    "train_data_path": r"sample_data.json",
    "valid_data_path": r"sample_data.json",
    "beam_size": 9,
    "bert_path": r"E:\BaiduNetdiskDownload\第六周 语言模型\bert-base-chinese",
    }

