# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "output",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "schema_path": "ner_data/schema.json",
    "max_length": 100,
    "num_layers": 3,
    "epoch": 10,
    "batch_size": 4,
    "tuning_tactics":"lora_tuning",
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "pretrain_model_path":r"D:\bert-base-chinese",
    "seed": 987
}