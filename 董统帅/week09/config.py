# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "model_type": "bert",
    "schema_path": "ner_data/schema.json",
    "train_data_path": "ner_data/train",
    "valid_data_path": "ner_data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 5,
    "optimizer": "adam",
    "learning_rate": 1e-3,
    "use_crf": True,
    "class_num": 10,
    "vocab_size": 21128,
    "bert_path": r"D:\Test\llm-cookbook-main\content\week06\bert-base-chinese"
}

