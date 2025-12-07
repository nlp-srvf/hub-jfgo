# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
import pandas as pd
import numpy as np
import os

"""
建立网络模型结构
"""


class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        class_num = config["class_num"]
        model_type = config["model_type"]
        num_layers = config["num_layers"]
        self.use_bert = False
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        if model_type == "fast_text":
            self.encoder = lambda x: x
        elif model_type == "lstm":
            self.encoder = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "gru":
            self.encoder = nn.GRU(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "rnn":
            self.encoder = nn.RNN(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif model_type == "cnn":
            self.encoder = CNN(config)
        elif model_type == "gated_cnn":
            self.encoder = GatedCNN(config)
        elif model_type == "stack_gated_cnn":
            self.encoder = StackGatedCNN(config)
        elif model_type == "rcnn":
            self.encoder = RCNN(config)
        elif model_type == "bert":
            self.use_bert = True
            self.encoder = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
            hidden_size = self.encoder.config.hidden_size
        elif model_type == "bert_lstm":
            self.use_bert = True
            self.encoder = BertLSTM(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_cnn":
            self.use_bert = True
            self.encoder = BertCNN(config)
            hidden_size = self.encoder.bert.config.hidden_size
        elif model_type == "bert_mid_layer":
            self.use_bert = True
            self.encoder = BertMidLayer(config)
            hidden_size = self.encoder.bert.config.hidden_size

        self.classify = nn.Linear(hidden_size, class_num)
        self.pooling_style = config["pooling_style"]
        self.loss = nn.functional.cross_entropy  # loss采用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        if self.use_bert:  # bert返回的结果是 (sequence_output, pooler_output)
            # sequence_output:batch_size, max_len, hidden_size
            # pooler_output:batch_size, hidden_size
            x = self.encoder(x)
        else:
            x = self.embedding(x)  # input shape:(batch_size, sen_len)
            x = self.encoder(x)  # input shape:(batch_size, sen_len, input_dim)

        if isinstance(x, tuple):  # RNN类的模型会同时返回隐单元向量，我们只取序列结果
            x = x[0]
        # 可以采用pooling的方式得到句向量
        if self.pooling_style == "max":
            self.pooling_layer = nn.MaxPool1d(x.shape[1])
        else:
            self.pooling_layer = nn.AvgPool1d(x.shape[1])
        x = self.pooling_layer(x.transpose(1, 2)).squeeze()  # input shape:(batch_size, sen_len, input_dim)

        # 也可以直接使用序列最后一个位置的向量
        # x = x[:, -1, :]
        predict = self.classify(x)  # input shape:(batch_size, input_dim)
        if target is not None:
            return self.loss(predict, target.squeeze())
        else:
            return predict


class CNN(nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        hidden_size = config["hidden_size"]
        kernel_size = config["kernel_size"]
        pad = int((kernel_size - 1) / 2)
        self.cnn = nn.Conv1d(hidden_size, hidden_size, kernel_size, bias=False, padding=pad)

    def forward(self, x):  # x : (batch_size, max_len, embeding_size)
        return self.cnn(x.transpose(1, 2)).transpose(1, 2)


class GatedCNN(nn.Module):
    def __init__(self, config):
        super(GatedCNN, self).__init__()
        self.cnn = CNN(config)
        self.gate = CNN(config)

    def forward(self, x):
        a = self.cnn(x)
        b = self.gate(x)
        b = torch.sigmoid(b)
        return torch.mul(a, b)


class StackGatedCNN(nn.Module):
    def __init__(self, config):
        super(StackGatedCNN, self).__init__()
        self.num_layers = config["num_layers"]
        self.hidden_size = config["hidden_size"]
        # ModuleList类内可以放置多个模型，取用时类似于一个列表
        self.gcnn_layers = nn.ModuleList(
            GatedCNN(config) for i in range(self.num_layers)
        )
        self.ff_liner_layers1 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.ff_liner_layers2 = nn.ModuleList(
            nn.Linear(self.hidden_size, self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_gcnn = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )
        self.bn_after_ff = nn.ModuleList(
            nn.LayerNorm(self.hidden_size) for i in range(self.num_layers)
        )

    def forward(self, x):
        # 仿照bert的transformer模型结构，将self-attention替换为gcnn
        for i in range(self.num_layers):
            gcnn_x = self.gcnn_layers[i](x)
            x = gcnn_x + x  # 通过gcnn+残差
            x = self.bn_after_gcnn[i](x)  # 之后bn
            # # 仿照feed-forward层，使用两个线性层
            l1 = self.ff_liner_layers1[i](x)  # 一层线性
            l1 = torch.relu(l1)  # 在bert中这里是gelu
            l2 = self.ff_liner_layers2[i](l1)  # 二层线性
            x = self.bn_after_ff[i](x + l2)  # 残差后过bn
        return x


class RCNN(nn.Module):
    def __init__(self, config):
        super(RCNN, self).__init__()
        hidden_size = config["hidden_size"]
        self.rnn = nn.RNN(hidden_size, hidden_size)
        self.cnn = GatedCNN(config)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.cnn(x)
        return x


class BertLSTM(nn.Module):
    def __init__(self, config):
        super(BertLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.rnn = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)

    def forward(self, x):
        x = self.bert(x)[0]
        x, _ = self.rnn(x)
        return x


class BertCNN(nn.Module):
    def __init__(self, config):
        super(BertCNN, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        config["hidden_size"] = self.bert.config.hidden_size
        self.cnn = CNN(config)

    def forward(self, x):
        x = self.bert(x)[0]
        x = self.cnn(x)
        return x


class BertMidLayer(nn.Module):
    def __init__(self, config):
        super(BertMidLayer, self).__init__()
        self.bert = BertModel.from_pretrained(config["pretrain_model_path"], return_dict=False)
        self.bert.config.output_hidden_states = True

    def forward(self, x):
        layer_states = self.bert(x)[2]  # (13, batch, len, hidden)
        layer_states = torch.add(layer_states[-2], layer_states[-1])
        return layer_states


# 优化器的选择
def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


def predict_on_test_set(model, test_loader, device):
    """
    在测试集上进行预测
    """
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch_data in test_loader:
            if device.type == 'cuda':
                batch_data = [d.cuda() for d in batch_data]

            input_ids, labels = batch_data
            outputs = model(input_ids)
            predictions = torch.argmax(outputs, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_predictions), np.array(all_labels)


def evaluate_predictions(predictions, labels):
    """
    评估预测结果
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted')
    recall = recall_score(labels, predictions, average='weighted')
    f1 = f1_score(labels, predictions, average='weighted')

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def compare_models(model_results):
    """
    对比不同模型的评估结果
    """
    df = pd.DataFrame(model_results)
    df = df.sort_values('accuracy', ascending=False)
    return df


if __name__ == "__main__":
    from config import Config
    from loader import load_data
    model_results = []

    # 测试不同的模型配置
    model_configs = [
        {"model_type": "lstm", "hidden_size": 128, "learning_rate": 1e-3},
        {"model_type": "cnn", "hidden_size": 128, "learning_rate": 1e-3},
        {"model_type": "gated_cnn", "hidden_size": 128, "learning_rate": 1e-3},
    ]

    # 加载测试数据
    test_data = load_data(Config["test_data_path"], Config)  # 假设有测试数据路径

    # 获取设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for config in model_configs:
        # 更新配置
        for key, value in config.items():
            Config[key] = value

        # 创建模型
        model = TorchModel(Config)

        # 加载预训练权重（如果有）
        model_path = f"./saved_models/{config['model_type']}_best.pth"
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"加载模型权重: {model_path}")

        model.to(device)

        # 在测试集上进行预测
        predictions, labels = predict_on_test_set(model, test_data, device)

        # 评估预测结果
        metrics = evaluate_predictions(predictions, labels)

        # 记录结果
        result = {
            'model_type': config['model_type'],
            'hidden_size': config['hidden_size'],
            'learning_rate': config['learning_rate']
        }
        result.update(metrics)
        model_results.append(result)

        print(f"模型 {config['model_type']} 在测试集上的评估结果:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        print()

    # 对比不同模型的结果
    comparison_df = compare_models(model_results)
    print("=== 模型对比结果 ===")
    print(comparison_df)

    # 保存结果到CSV文件
    comparison_df.to_csv("model_comparison_results.csv", index=False)
    print("模型对比结果已保存到 model_comparison_results.csv")

    # 找到最佳模型
    best_model = comparison_df.iloc[0]
    print(f"\n=== 最佳模型 ===")
    print(f"模型类型: {best_model['model_type']}")
    print(f"隐藏层大小: {best_model['hidden_size']}")
    print(f"学习率: {best_model['learning_rate']}")
    print(f"准确率: {best_model['accuracy']:.4f}")
    print(f"F1分数: {best_model['f1_score']:.4f}")
