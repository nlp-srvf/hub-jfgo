import torch
import torch.nn as nn
import numpy as np
import random
import string

"""

基于pytorch框架编写模型训练
使用RNN实现找出字符串中指定字符的位置
例如：
    找出a出现的位置
    abcde --->  0
    forwa --->  4
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # 定义embedding层
        self.layer = nn.RNN(vector_dim, 20, bias=False, batch_first=True)  # 定义RNN层
        self.classifier = nn.Linear(vector_dim, 5)  # 线性层
        self.loss = nn.functional.cross_entropy  # loss 函数交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)
        output, hidden = self.layer(x)
        hidden = hidden.squeeze(0)
        y_pred = self.classifier(hidden)
        if y is not None:
            return self.loss(y_pred, y)
        else:
            return y_pred


#创建词表
def build_vocab():
    chars = string.ascii_lowercase + ' '
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字符串对应一个序号，从1开始
    vocab['unk'] = len(chars)
    return vocab

#创建训练样本
def build_sample(vocab, str_length):
    chars = string.ascii_lowercase + ' '
    x = [random.choice(chars) for _ in range(str_length)]  # 随机从词表选择指定长度字符串，徐允许重复
    if 'a' not in x:
        x[random.randint(0,len(x) - 1)] = 'a'
    y = x.index('a')
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字符转换序号，为embedding做准备
    return x, y


#创建训练样本数据集
def build_dataset(sample_length, vocab, str_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, str_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#构建模型
def build_model(vocab, vector_dim):
    return TorchModel(vector_dim, vocab)

#预测值对比
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)  # 建立200个用于测试的样本
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    vector_dim = 20  # 每个字的维度
    str_length = 5  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, vector_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, str_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, str_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    return

#测试
def predict(model_path, input_strings):
    char_dim = 20  # 每个字的维度
    vocab = build_vocab()
    model = build_model(vocab, char_dim)     #建立模型
    model.load_state_dict(torch.load(model_path,weights_only=False))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        x.append([vocab[char] for char in input_string])  #将输入序列化
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(torch.LongTensor(x))  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d" % (input_string, round(float(torch.argmax(result[i]))))) #打印结果


if __name__ == "__main__":
    # main()
    # vocab = build_vocab()
    # for i in range(10):
    #     x,y = build_sample(vocab,5)
    #     print("x ==> {},y ==> {}".format(x,y))
    test_strings = ["abcde", "ildka", "dlman", "lalnk"]
    predict("model.pth", test_strings)
