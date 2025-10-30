import random

import torch
import torch.nn as nn
import numpy as np

"""
尝试在nlpdemo中使用rnn模型训练，判断特定字符在文本中的位置。

判断特定长度字符串在文本中的起始位置
难点： 神经网络结构设计 符合维度，考虑合理性 快速收敛
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super().__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # self.pool = nn.AvgPool1d(sentence_length)
        self.layer = nn.RNN(vector_dim, vector_dim * 3, batch_first=True)
        self.linear = nn.Linear(vector_dim*3, len(vocab))
        # self.activation = torch.sigmoid
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        y_pred = self.embedding(x)  # [50, 3]  --> [50, 3, 8]
        # y_pred = y_pred.transpose(1, 2)
        # y_pred = self.pool(y_pred).squeeze()  # [50, 8]
        y_pred, _ = self.layer(y_pred)
        y_pred = self.linear(y_pred.mean(dim=1))
        # input(y_pred.shape)
        # y_pred = self.activation(y_pred)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return torch.argmax(y_pred, dim=1)


def build_vocab():
    chars = 'abcdefghijklmnopqrstuvwxyz'
    vocab = {}
    for i, char in enumerate(chars):
        vocab[char] = i
    vocab['unk'] = len(vocab)
    print(vocab)
    return vocab


def build_sample(vocab, sentence_length):
    # input(vocab.keys())
    chars = [char for char in vocab.keys()]
    start_idx = random.randint(1, len(vocab))
    if sentence_length + start_idx <= len(vocab):
        char = chars[start_idx:start_idx + sentence_length]
    else:
        char = chars[start_idx // 2:start_idx // 2 + sentence_length]
    # char = random.choices(list(vocab.keys()))
    y = vocab[char[0]]
    x = [vocab.get(ch, vocab['unk']) for ch in char]
    # print(char, x, y)
    return x, y


def build_dataset(vocab, sample_num, sentence_length):
    dataset_x, dataset_y = [], []
    for sample in range(sample_num):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


def evaluate(model, vocab):
    model.eval()
    val_sample_num = 1000
    sentence_length = 3
    with torch.no_grad():
        x, y = build_dataset(vocab, val_sample_num, sentence_length)
        y_pred = model(x)
        # correct = 0
        # for yp, yt in zip(y_pred, y):
        #     if int(yp) == int(yt):
        #         correct += 1
        acc = (y_pred == y).float().mean().item()
        # print(f'准确率为： {acc}')
    return acc


def main():
    epoches = 10
    batch_size = 20
    train_sample_num = 500
    learning_rate = 1e-2
    vector_dim = 10
    sentence_length = 3

    vocab = build_vocab()
    model = TorchModel(vector_dim, sentence_length, vocab)
    optim = torch.optim.Adam(model.parameters(), learning_rate)

    for epoch in range(epoches):
        model.train()
        batch_loss = []
        for batch in range(train_sample_num // batch_size):
            x, y = build_dataset(vocab, batch_size, sentence_length)
            loss = model(x, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            batch_loss.append(loss.item())
        acc = evaluate(model, vocab)
        # input(batch_loss)
        print(f'epoch: {epoch} 平均loss为: {np.mean(batch_loss)}, acc为: {acc}')
    return


if __name__ == '__main__':
    main()
