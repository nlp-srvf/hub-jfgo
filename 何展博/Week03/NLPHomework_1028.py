# coding:utf8

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json

"""
基于pytorch的网络编写
实现一个多分类任务：返回文本中'我'的索引位置
文本长度固定为10，如果'我'存在多个返回所有位置，不存在返回10
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.LSTM(vector_dim, vector_dim, batch_first=True)  # LSTM层
        self.classify = nn.Linear(vector_dim, 2)  # 2个输出（0:不是我，1:是我）
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # LSTM处理序列
        # 重塑为 (batch_size * sen_len, vector_dim)
        batch_size, sen_len, vector_dim = x.shape
       # x = x.view(batch_size * sen_len, vector_dim)
        x = x.reshape(batch_size * sen_len, vector_dim)

        x = self.classify(x)  # (batch_size * sen_len, 2)

        if y is not None:
            # 将标签转换为类别索引 (0或1)
            y = y.view(-1)  # (batch_size * sen_len)
            return self.loss(x, y.long())  # 计算交叉熵损失
        else:
            # 预测：对每个位置计算softmax，取类别1的概率
            prob = F.softmax(x, dim=1)  # (batch_size * sen_len, 2)
            # 重塑回 (batch_size, sen_len)
            # return prob[:, 1].view(batch_size, sen_len)  # (batch_size, sen_len)
            return prob[:, 1].reshape(batch_size, sen_len)


# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 26
    return vocab


# 随机生成一个样本
# 从所有字中选取sentence_length个字
# 标签：10维向量，如果该位置是'我'则为1，否则为0
def build_sample(vocab, sentence_length=10):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 构建标签：10维向量，如果该位置是'我'则为1，否则为0
    y = [1 if char == '我' else 0 for char in x]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, y


# 建立数据集
# 输入需要的样本数量
def build_dataset(sample_length, vocab, sentence_length=10):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # y是10维向量
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length=10):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 (batch, 10)
        for i in range(len(y_pred)):
            # 将预测结果转换为0/1（>0.5为1，否则为0）
            pred_positions = [idx for idx, prob in enumerate(y_pred[i]) if prob >= 0.5]
            # 获取真实位置
            real_positions = [idx for idx, val in enumerate(y[i]) if val == 1]

            # 比较预测位置和真实位置
            if pred_positions == real_positions:
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度（固定为10）
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 10  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    # 修复警告：显式设置 weights_only=True
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重
    x = []
    for input_string in input_strings:
        # 确保输入字符串长度为10
        if len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]
        else:
            # 补充到长度10
            input_string = input_string + ' ' * (sentence_length - len(input_string))

        # 将输入序列化
        x.append([vocab.get(char, vocab['unk']) for char in input_string])

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测 (batch, 10)

    for i, input_string in enumerate(input_strings):
        # 获取预测位置
        positions = [idx for idx, prob in enumerate(result[i]) if prob >= 0.5]
        # 如果没有找到'我'，返回10
        if not positions:
            print("输入：%s, 预测位置：10" % input_string)
        else:
            print("输入：%s, 预测位置：%s" % (input_string, str(positions)))


if __name__ == "__main__":
    main()
    # 测试用例
    # test_strings = ["我爱中国", "hello我", "world", "我我我", "abcde"]
    # predict("model.pth", "vocab.json", test_strings)