# -*- coding: utf-8 -*-
"""
配置文件：包含所有模型的超参数和路径设置
"""
import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(BASE_DIR), "文本分类练习.csv")
MODEL_DIR = os.path.join(BASE_DIR, "saved_models")
VOCAB_PATH = os.path.join(BASE_DIR, "vocab.json")

# 确保模型保存目录存在
os.makedirs(MODEL_DIR, exist_ok=True)

# 数据集划分比例
TRAIN_RATIO = 0.7
VALID_RATIO = 0.15
TEST_RATIO = 0.15

# 随机种子
RANDOM_SEED = 42

# 通用参数
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 64

# SVM参数
SVM_CONFIG = {
    "C": 1.0,
    "kernel": "linear",
    "max_features": 5000
}

# RNN参数
RNN_CONFIG = {
    "vocab_size": MAX_VOCAB_SIZE,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 1,
    "dropout": 0.3,
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": BATCH_SIZE
}

# LSTM参数
LSTM_CONFIG = {
    "vocab_size": MAX_VOCAB_SIZE,
    "embedding_dim": 128,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "learning_rate": 0.001,
    "epochs": 10,
    "batch_size": BATCH_SIZE
}

# BERT参数
BERT_CONFIG = {
    "model_name": "bert-base-chinese",
    "max_length": 128,
    "learning_rate": 2e-5,
    "epochs": 3,
    "batch_size": 16
}

# 标签映射
LABEL_MAP = {
    0: "差评",
    1: "好评"
}

