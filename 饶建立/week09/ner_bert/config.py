# -*- coding: utf-8 -*-

"""
配置参数信息 - BERT版本
"""


Config = {
    "model_path": "model_output",
    "schema_path": "../ner_data/schema.json",
    "train_data_path": "../ner_data/train",
    "valid_data_path": "../ner_data/test",
    "max_length": 128,
    "hidden_size": 768,  # BERT base hidden size
    "epoch": 10,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 2e-5,  # BERT通常使用较小的学习率
    "use_crf": True,
    "class_num": 9,
    "bert_path": r"E:\pretrain_models\bert-base-chinese"  # BERT预训练模型路径
}

