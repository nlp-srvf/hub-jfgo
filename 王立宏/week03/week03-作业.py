# coding:utf8

"""
NlpDemo中使用rnn模型训练，判断特定字符在文本中的位置
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os

"""

基于pytorch的RNN网络编写
实现一个序列标注任务，判断文本中特定字符的位置
使用RNN模型进行五分类（对应5个位置）

"""


class MyRnnModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=128, num_layers=2, num_classes=5):
        """
        :param vector_dim: 每个字的维度
        :param sentence_length: 每个序列的长度
        :param vocab: 词表
        :param hidden_size: 隐藏层维度
        :param num_layers: RNN层数
        :param num_classes: 分类数量（位置类别）
        """
        super(MyRnnModel, self).__init__()
        self.vector_dim = vector_dim
        self.sentence_length = sentence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        # embedding层
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # RNN层
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入形状为(batch, seq, feature)
            bidirectional=False  # 单向RNN
        )
        # 线性层
        self.classifier = nn.Linear(hidden_size, num_classes)
        # 使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        """
        前向传播
        :param x: 输入张量，形状为(batch_size, sentence_length)
        :param y: 真实标签，形状为(batch_size,)
        :return: 预测结果或损失值
        """
        batch_size = x.size(0)
        # embedding: (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        x = self.embedding(x)
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size)
        # RNN前向传播: (batch_size, sentence_length, vector_dim) ->
        # output: (batch_size, sentence_length, hidden_size)
        # hn: (num_layers, batch_size, hidden_size)
        output, hn = self.rnn(x, h0)
        # 取最后一个时间步的输出作为整个序列的表示
        # output: (batch_size, sentence_length, hidden_size) -> (batch_size, hidden_size)
        last_output = output[:, -1, :]
        # 分类: (batch_size, hidden_size) -> (batch_size, num_classes)
        logits = self.classifier(last_output)
        if y is not None:
            # 计算交叉熵损失
            return self.loss(logits, y.long())
        else:
            return logits


# 字符集
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 判断特定字符"你我他"在文本中的位置（五分类）
    # 0: 没有特定字符
    # 1: 在位置0
    # 2: 在位置1
    # 3: 在位置2
    # 4: 在位置3或之后
    y = 0  # 默认没有特定字符
    for i, char in enumerate(x):
        if char in "你我他":
            if i == 0:
                y = 1
            elif i == 1:
                y = 2
            elif i == 2:
                y = 3
            else:
                y = 4
            break  # 只找第一个特定字符的位置
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def main(model_path, vocab_path):
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 2000  # 每轮训练总共训练的样本总数
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    hidden_size = 128  # RNN隐藏层维度
    num_layers = 2  # RNN层数
    num_classes = 5  # 分类数量（0-4）
    learning_rate = 0.001  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = MyRnnModel(char_dim, sentence_length, vocab, hidden_size, num_layers, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        # 每5轮评估一次
        if (epoch + 1) % 5 == 0:
            acc = evaluate(model, vocab, sentence_length)
            log.append([acc, avg_loss])
    # 保存模型
    torch.save(model.state_dict(), model_path)
    # 保存词表
    with open(vocab_path, "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    print("训练完成，模型已保存!")
    return log


# 评估函数
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    # 统计各类别样本数量
    class_counts = [torch.sum(y == i).item() for i in range(5)]
    print("各类别样本数量:", class_counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        predictions = torch.argmax(y_pred, dim=1)
        correct = (predictions == y).sum().item()
        wrong = len(y) - correct
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


# 预测函数
def predict(model_path, vocab_path, input_strings):
    char_dim = 64
    sentence_length = 6
    hidden_size = 128
    num_layers = 2
    num_classes = 5
    # 加载词表
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    # 建立模型
    model = MyRnnModel(char_dim, sentence_length, vocab, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # 处理输入
    x = []
    for input_string in input_strings:
        # 填充或截断到固定长度
        if len(input_string) < sentence_length:
            input_string += 'pad' * (sentence_length - len(input_string))
        else:
            input_string = input_string[:sentence_length]
        x.append([vocab.get(char, vocab['unk']) for char in input_string])
    # 预测
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        probabilities = torch.softmax(result, dim=1)
        predictions = torch.argmax(result, dim=1)
    # 输出结果
    position_map = {
        0: "无特定字符",
        1: "位置0",
        2: "位置1",
        3: "位置2",
        4: "位置3或之后"
    }
    print("==================== 预测结果 ====================")
    for i, (input_str, pred, prob) in enumerate(zip(input_strings, predictions, probabilities)):
        actual_pos = -1
        for j, char in enumerate(input_str):
            if char in "你我他":
                actual_pos = j
                break
        actual_label = 0  # 无特定字符
        if actual_pos >= 0:
            if actual_pos == 0:
                actual_label = 1
            elif actual_pos == 1:
                actual_label = 2
            elif actual_pos == 2:
                actual_label = 3
            else:
                actual_label = 4
        is_correct = "✓" if pred.item() == actual_label else "✗"
        confidence = prob[pred.item()].item()
        print(f"输入: {input_str}")
        print(f"  预测: {position_map[pred.item()]} (置信度: {confidence:.4f}) {is_correct}")
        print(f"  实际: {position_map[actual_label]}")
        print("-" * 50)


if __name__ == "__main__":
    os.makedirs("tmp/model", exist_ok=True)
    model_path = "tmp/model/MyRnnModel.pt"
    vocab_path = "tmp/model/vocab.json"
    # 训练模型
    log = main(model_path, vocab_path)
    # 测试预测
    test_strings = [
        "fnvf我e",  # 特定字符在位置4 → 类别4
        "wz你dfg",  # 特定字符在位置2 → 类别2
        "rqwdeg",  # 无特定字符 → 类别0
        "n我kwww",  # 特定字符在位置1 → 类别1
        "他abcde",  # 特定字符在位置0 → 类别1
        "ab你cde"  # 特定字符在位置2 → 类别2
    ]
    predict(model_path, vocab_path, test_strings)
