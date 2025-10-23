# coding:utf8

import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt

"""
基于pytorch框架实现多分类任务
规律：x是一个5维向量，最大值所在的维度即为其类别（0-4）
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 线性层输出类别数
        self.activation = nn.Softmax(dim=1)  # softmax将输出转为概率分布
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        y_pred = self.activation(x)  # 计算概率分布
        if y is not None:
            # 交叉熵损失需要原始logits输入，且标签为长整型
            return self.loss(x, y.long())
        else:
            return y_pred  # 输出预测概率


# 生成一个样本, 最大值所在维度为类别，数据范围扩展到-10到10之间
def build_sample():
    # 生成-10到10之间的随机数，扩大数据范围
    x = np.random.uniform(low=-10.0, high=10.0, size=5)
    # 找到最大值所在的索引（0-4）作为标签
    label = np.argmax(x)
    return x, label


# 随机生成一批样本，优化数据转换效率
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # 先转换为numpy数组再转tensor，解决性能警告
    return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(Y))


# 测试模型准确率
def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    print(f"测试集中各类别样本数：{np.bincount(y.long().numpy())}")

    correct = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        # 取概率最大的类别作为预测结果
        pred_labels = torch.argmax(y_pred, dim=1)
        correct = (pred_labels == y.long()).sum().item()

    acc = correct / test_sample_num
    print(f"正确预测个数：{correct}, 正确率：{acc:.4f}")
    return acc


def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总样本数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数（0-4）
    learning_rate = 0.01  # 学习率

    # 建立模型
    model = TorchModel(input_size, num_classes)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 创建训练集
    train_x, train_y = build_dataset(train_sample)

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 取出一个batch的数据
            x = train_x[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size: (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算损失
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss:{np.mean(watch_loss):.6f}")
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model.bin")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.title("Training Accuracy and Loss")
    plt.xlabel("Epoch")
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    # 加载模型时指定weights_only=True，解决安全警告
    model.load_state_dict(torch.load(model_path, weights_only=True))  # 加载训练好的权重

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model(torch.FloatTensor(input_vec))  # 模型预测

    for vec, res in zip(input_vec, result):
        pred_label = torch.argmax(res).item()
        max_value = np.max(vec)
        true_label = np.argmax(vec)
        print(f"输入：{vec}, 最大值位置：{true_label}, 预测位置：{pred_label}, 概率：{res[pred_label]:.4f}")


if __name__ == "__main__":
    main()
    # 测试示例（包含正负值和更大范围的数据）
    test_vec = [
        [1.1, -2.2, 3.3, -4.4, 5.5],  # 最大值在4
        [10.5, 3.2, -2.1, -1.8, 4.3],  # 最大值在0
        [-2.2, 9.1, 3.5, 5.7, 1.2],  # 最大值在1
        [8.3, -2.4, 7.1, 6.5, -5.2]  # 最大值在0
    ]
    predict("multi_class_model.bin", test_vec)