# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()
        hidden_size = config["hidden_size"]
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.layer = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        x = self.layer(x)
        x = nn.functional.max_pool1d(x.transpose(1, 2), x.shape[1]).squeeze()
        return x


class SiameseNetwork(nn.Module):
    def __init__(self, config):
        super(SiameseNetwork, self).__init__()
        self.sentence_encoder = SentenceEncoder(config)
        # 设置默认的margin值，用于三元组损失
        self.margin = config.get("triplet_margin", 0.5)

    def cosine_distance(self, tensor1, tensor2):
        """计算余弦距离：1 - cosine_similarity"""
        tensor1 = torch.nn.functional.normalize(tensor1, dim=-1)
        tensor2 = torch.nn.functional.normalize(tensor2, dim=-1)
        cosine_sim = torch.sum(torch.mul(tensor1, tensor2), dim=-1)
        return 1 - cosine_sim

    def cosine_triplet_loss(self, anchor, positive, negative, margin=None):
        """
        计算基于余弦距离的三元组损失
        训练时使用
        """
        if margin is None:
            margin = self.margin

        pos_distance = self.cosine_distance(anchor, positive)
        neg_distance = self.cosine_distance(anchor, negative)

        losses = torch.relu(pos_distance - neg_distance + margin)
        return losses.mean()

    def forward(self, sentence1, sentence2=None, sentence3=None, mode="train", margin=None):
        """
        修改后的前向传播，支持不同模式
        参数:
            sentence1: 第一个句子
            sentence2: 第二个句子（训练时为positive，预测时为第二个句子）
            sentence3: 第三个句子（仅训练时需要，作为negative）
            mode: 模式，"train"或"predict"
            margin: 三元组损失的margin（仅训练时使用）
        """
        if mode == "train":
            # 训练模式：需要三个句子，计算三元组损失
            if sentence3 is None:
                raise ValueError("训练模式需要三个句子：anchor, positive, negative")
            # 编码三个句子
            anchor_vec = self.sentence_encoder(sentence1)  # anchor
            positive_vec = self.sentence_encoder(sentence2)  # positive
            negative_vec = self.sentence_encoder(sentence3)  # negative
            # 计算三元组损失
            loss = self.cosine_triplet_loss(anchor_vec, positive_vec, negative_vec, margin)
            return loss
        elif mode == "predict":
            # 预测模式：需要两个句子，返回相似度
            if sentence2 is None:
                raise ValueError("预测模式需要两个句子")
            # 编码两个句子
            vec1 = self.sentence_encoder(sentence1)
            vec2 = self.sentence_encoder(sentence2)
            # 计算余弦相似度
            similarity = self.cosine_similarity(vec1, vec2)
            return similarity
        else:
            raise ValueError("模式必须为'train'或'predict'")

    def encode_sentence(self, sentence):
        """编码单个句子，返回向量表示"""
        return self.sentence_encoder(sentence)


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

# 预测函数示例
def predict_similarity(model, sentence1, sentence2, device="cpu"):
    """预测两个句子的相似度"""
    model.eval()
    with torch.no_grad():
        # 确保输入是批量的形式
        if len(sentence1.shape) == 1:
            sentence1 = sentence1.unsqueeze(0)
        if len(sentence2.shape) == 1:
            sentence2 = sentence2.unsqueeze(0)
        sentence1, sentence2 = sentence1.to(device), sentence2.to(device)
        # 计算相似度
        similarity = model(sentence1, sentence2, mode="predict")
        return similarity


# 测试代码
if __name__ == "__main__":
    from config import Config
    Config["vocab_size"] = 10
    Config["max_length"] = 4
    model = SiameseNetwork(Config)
    s1 = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]])
    s2 = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]])
    l = torch.LongTensor([[1], [0]])
    y = model(s1, s2, l)
    print(y)
