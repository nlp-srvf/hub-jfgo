# coding:utf8
# 解决OpenMP库冲突问题
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""
基于pytorch框架编写模型训练
实现一个多分类任务：输入一个5维向量，找出最大值所在的维度作为类别标签
类别：0,1,2,3,4 分别对应第1,2,3,4,5个维度的最大值
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 线性层，输出output_size个类别
        self.activation = nn.Softmax(dim=1)  # Softmax激活函数
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值，用于区分训练阶段和推理阶段
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size,input_size) -> (batch_size,output_size)
        if y is not None:
            return self.loss(x, y)  # 计算交叉熵损失 训练阶段
        else:
            return self.activation(x)  # 返回各类别的概率分布 推理阶段


# 随机生成一个5维向量，找出最大值所在的位置作为类别标签
def build_sample():
    x = np.random.random(5)  # 随机生成一个包含5个元素的数组
    max_index = np.argmax(x)  # 找出最大值的索引作为类别标签
    return x, max_index


def build_dataset(total_sample_num):
    X = []  # 定义空list
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 转换成张量


# 测试代码：测试模型每轮训练的准确率
def evaluate(model):
    model.eval()  # 模型切换到评估模式
    test_sample_num = 1000  # 测试样本数量
    x, y = build_dataset(test_sample_num)

    correct, wrong = 0, 0
    with torch.no_grad():  # 临时关闭梯度计算
        y_pred = model(x)  # 模型预测
        pred_classes = torch.argmax(y_pred, dim=1)  # 获取预测的类别
        # 对比预测类别和真实类别
        for pred_class, true_class in zip(pred_classes, y):
            if pred_class.item() == true_class.item():
                correct += 1
            else:
                wrong += 1

        accuracy = correct / (correct + wrong)
        print(f"本次预测集中共有{test_sample_num}个样本，正确预测个数：{correct},准确率：{accuracy}")
        return accuracy


def main():
    # 配置参数
    epoch_num = 50  # 训练轮数
    batch_size = 32  # 一次训练样本个数 -- 批次大小
    train_sample_num = 10000  # 训练样本总数
    input_size = 5  # 输入向量维度（特征数）
    output_size = 5  # 输出特征数 -- 5个类别（0,1,2,3,4）\
    learning_rate = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size, output_size)
    print("================================== 模型结构 =====================================\n")
    print(model)

    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # 创建训练集
    train_x, train_y = build_dataset(train_sample_num)
    print("================================== 训练集 =====================================\n")
    print(f"X:{train_x}\nY:{train_y}")

    # 训练
    print("================================== 模型训练 =====================================\n")
    log = []  # 定义列表记录准确率、损失值
    for epoch in range(epoch_num):
        model.train()  # 模型切换到训练模式
        watch_loss = []  # 定义列表记录损失值

        for batch_index in range(0, train_sample_num, batch_size):
            # 取出一个batch -- 一次训练样本个数
            x = train_x[batch_index:batch_index + batch_size]
            y = train_y[batch_index:batch_index + batch_size]
            # 如果样本量不足一个batch大小，跳过
            if len(x) < batch_size:
                break

            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算损失函数梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())  # loss通常是张量，loss.item()转换为标量，用于数值分析

        avg_loss = np.mean(watch_loss)
        print("=================================================================\n第%d轮平均loss: %f" % (epoch + 1, avg_loss))
        acc = evaluate(model)  # 测试本轮模型训练结果
        log.append([acc, avg_loss])

    # 保存模型
    torch.save(model.state_dict(), "multi_class_model_zft.bin")
    print("=================================================================\n")
    print("模型已保存为 multi_class_model_zft.bin")

    # 画图
    print("训练日志:", log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return model


# 模型验证测试
def predict(model_path, input_vec):
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path))  # 加载模型
    print(f"模型权重：{model.state_dict()}")
    print("-" * 100)

    model.eval()  # 模型切换到评估模式
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))

    for i, (vec, res) in enumerate(zip(input_vec, result)):
        pred_class = torch.argmax(res).item()
        max_value = torch.max(torch.FloatTensor(vec)).item()
        print(f"输入样本{i + 1}:{vec}")
        print(f"最大值：{max_value} (位置：{pred_class})")
        print(f"各类别概率：{res}")
        print("-" * 100)


if __name__ == "__main__":
    # 训练
    main()
    # 验证
    test_vec = [
        [0.1, 0.8, 0.3, 0.2, 0.5],  # 最大值在位置1
        [0.9, 0.2, 0.1, 0.3, 0.4],  # 最大值在位置0
        [0.2, 0.3, 0.7, 0.1, 0.6],  # 最大值在位置2
        [0.4, 0.5, 0.2, 0.8, 0.1],  # 最大值在位置3
        [0.3, 0.2, 0.1, 0.4, 0.9]  # 最大值在位置4
    ]

    print("\n================================== 验证测试 =====================================")
    predict("multi_class_model_zft.bin", test_vec)
