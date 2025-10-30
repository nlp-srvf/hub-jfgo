# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的RNN网络
判断文本中特定字符（你、我、他）的位置
"""


class TorchRNNModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=16, num_classes=2):
        super(TorchRNNModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.RNN(input_size=vector_dim,
                          hidden_size=hidden_size,
                          batch_first=True,
                          num_layers=1)  # RNN层
        self.classify = nn.Linear(hidden_size, num_classes)  # 分类层，输出是否为特定字符
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失，适合多分类

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sen_len, hidden_size)
        y_pred = self.classify(rnn_out)  # (batch_size, sen_len, num_classes)

        if y is not None:
            # 计算每个位置的损失并平均
            return self.loss(y_pred.transpose(1, 2), y)
        else:
            return y_pred  # 返回每个位置的预测结果


# 构建字符集
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 生成包含特定字符位置标注的数据
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 构建标签：1表示是特定字符（你/我/他），0表示不是
    y = [1 if word in {"你", "我", "他"} else 0 for word in x]

    # 将字转换成序号
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


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchRNNModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, total = 0, 0

    with torch.no_grad():
        y_pred = model(x)  # (batch_size, sen_len, num_classes)
        y_pred = torch.argmax(y_pred, dim=2)  # 获取每个位置的预测类别

        # 计算每个位置的准确率
        for batch in range(y.shape[0]):
            for pos in range(y.shape[1]):
                if y_pred[batch, pos] == y[batch, pos]:
                    correct += 1
                total += 1

    acc = correct / total
    print(f"正确预测个数：{correct}, 总预测个数：{total}, 正确率：{acc:.4f}")
    return acc


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
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
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "rnn_model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    x = []
    for input_string in input_strings:
        # 截断或补长到固定长度
        if len(input_string) > sentence_length:
            input_string = input_string[:sentence_length]
        else:
            input_string += "pad" * (sentence_length - len(input_string))
        x.append([vocab.get(char, vocab['unk']) for char in input_string])

    with torch.no_grad():
        result = model(torch.LongTensor(x))  # (batch_size, sen_len, num_classes)
        pred = torch.argmax(result, dim=2)  # 获取每个位置的预测结果

    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}")
        print(f"字符位置预测（1表示特定字符，0表示普通字符）：{pred[i].numpy().tolist()}")
        print("---")


if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww", "你他我abc"]
    predict("rnn_model.pth", "vocab.json", test_strings)
