# coding: utf-8

'''
config.py：输入模型配置参数，如学习率、模型保存位置等
'''

Config = {
    "model_path": "model_output",
    "model_type": "bert",
    "schema_path": "./data/schema.json",
    "train_data_path": "./data/train",
    "valid_data_path": "./data/test",
    "vocab_path":"chars.txt",
    "max_length": 100,
    "hidden_size": 256,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 5,
    "optimizer": "adamw",
    "learning_rate": 2e-5,
    "use_crf": True,
    "class_num": 10,
    "vocab_size": 21128,
    "bert_path": r"./bert-base-chinese"
}