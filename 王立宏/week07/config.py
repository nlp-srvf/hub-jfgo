# coding: utf-8

'''
config.py：输入模型配置参数，如学习率、模型保存位置等
'''

import torch
import os

class Config:
    def __init__(self):
        # 数据路径
        self.data_path = "data/文本分类练习.csv"
        self.model_save_path = "./model"
        self.result_path = "./result.csv"

        # 通用参数
        self.random_seed = 42
        self.batch_size = 64
        self.max_len = 128
        self.test_size = 0.2
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # BERT模型参数
        self.bert_model_name = "bert-base-chinese"
        self.bert_local_path = "./bert-base-chinese"  # 本地BERT模型路径
        self.bert_tokenizer_path = "./bert-base-chinese"  # 本地BERT tokenizer路径
        self.bert_lr = 2e-5
        self.bert_epochs = 3
        self.bert_saved_path = os.path.join(self.model_save_path, "bert_model.pth")

        # FastText参数
        self.fasttext_lr = 0.1
        self.fasttext_epochs = 50
        self.fasttext_word_ngrams = 2
        self.fasttext_saved_path = os.path.join(self.model_save_path, "fasttext_model.bin")

        # TextCNN参数
        self.textcnn_lr = 1e-3
        self.textcnn_epochs = 10
        self.textcnn_embedding_dim = 300
        self.textcnn_num_filters = 100
        self.textcnn_filter_sizes = [2, 3, 4]
        self.textcnn_dropout = 0.5
        self.textcnn_saved_path = os.path.join(self.model_save_path, "textcnn_model.pth")

        # LSTM参数
        self.lstm_lr = 1e-3
        self.lstm_epochs = 10
        self.lstm_hidden_size = 128
        self.lstm_num_layers = 2
        self.lstm_dropout = 0.5
        self.lstm_saved_path = os.path.join(self.model_save_path, "lstm_model.pth")

        # 门控CNN参数
        self.gatedcnn_lr = 1e-3
        self.gatedcnn_epochs = 10
        self.gatedcnn_embedding_dim = 300
        self.gatedcnn_num_filters = 100
        self.gatedcnn_kernel_size = 3
        self.gatedcnn_saved_path = os.path.join(self.model_save_path, "gated_cnn_model.pth")

        # 朴素贝叶斯参数
        self.nb_saved_path = os.path.join(self.model_save_path, "naive_bayes_model.pkl")

        # SVM参数
        self.svm_saved_path = os.path.join(self.model_save_path, "svm_model.pkl")
        self.svm_c = 1.0
        self.svm_kernel = 'rbf'

        # 创建模型保存目录
        os.makedirs(self.model_save_path, exist_ok=True)

        # 多次重复评估配置
        self.repeat_evaluation_times = 10  # 每个模型重复训练测试的次数