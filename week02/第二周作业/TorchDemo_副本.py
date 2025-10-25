# coding:utf8

# 解决 OpenMP 库冲突问题
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
实现一个自行构造的找规律(机器学习)任务 - 多分类版本
规律：x是一个5维向量，最大值所在的索引，即为样本的类别 (0-4)
"""


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes): # 增加类别数量参数
        super(TorchModel, self).__init__()
        # 1. 修改输出维度为类别数 num_classes
        self.linear = nn.Linear(input_size, num_classes)
        # 2. 移除 activation，因为 CrossEntropyLoss 会自动应用 Softmax
        # self.activation = torch.sigmoid # 不再需要
        # 3. 修改损失函数为交叉熵损失
        self.loss = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        # 4. 直接获取线性层的输出 (logits)
        logits = self.linear(x)  # (batch_size, input_size) -> (batch_size, num_classes)
        if y is not None:
            # 5. CrossEntropyLoss 期望的 y 是类别索引，需要是 LongTensor 类型
            return self.loss(logits, y)
        else:
            # 在预测时可以手动加上softmax来获取概率分布
            return torch.softmax(logits, dim=-1)


# 生成一个样本
# 规律：五维向量中最大值所在的索引，即为该样本的类别
def build_sample():
    x = np.random.random(5)
    # 使用 np.argmax 获取最大值所在的索引
    y = np.argmax(x)
    return x, y


# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        # 6. Y 现在是一维的类别索引列表
        Y.append(y)
    # Y 需要是 Torch.LongTensor 类型以匹配 CrossEntropyLoss 的要求
    return torch.FloatTensor(X), torch.LongTensor(Y)


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200 # 增加测试样本数量以便观察
    x, y = build_dataset(test_sample_num) # x: (200, 5), y: (200)

    # 统计各类别的样本数量
    class_counts = {i: 0 for i in range(model.linear.out_features)}
    for label in y:
       class_counts[int(label.item())] += 1
    print("本次预测集中各类样本数量:", class_counts)

    correct, wrong = 0, 0
    with torch.no_grad():
        # 7. 模型输出的是概率分布或logits，需要找到最大值索引
        y_pred_probs = model(x)  # 模型预测, shape: (test_sample_num, num_classes)
        # 使用 argmax 获取预测的类别
        predicted_classes = torch.argmax(y_pred_probs, dim=1) # shape: (test_sample_num)

        # 8. 直接比较预测类别和真实类别
        correct = (predicted_classes == y).sum().item()
        wrong = len(y) - correct

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    num_classes = 5  # 新增：类别数量
    learning_rate = 0.001
    
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
        # 注意：train_y现在是一维的，所以下面的索引方式依然有效
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            # 9. y 张量的形状是 (batch_size), 不需要再展平或修改
            loss = model(x, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])
        
    # 保存模型
    torch.save(model.state_dict(), "model_multi_class.bin")
    
    # 画图
    print("Log (accuracy, loss):", log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    # print(model.state_dict()) # 可以取消注释查看模型权重

    model.eval()
    with torch.no_grad():
        results = model.forward(torch.FloatTensor(input_vec))
        
    for vec, res in zip(input_vec, results):
        # 10. 使用 argmax 获取预测类别
        predicted_class = torch.argmax(res).item()
        print("输入：%s, 预测类别：%d, 概率分布：%s" % (np.array(vec), predicted_class, res.numpy()))


if __name__ == "__main__":
    main()
    
    # 创建一些测试向量
    test_vec = [[0.1, 0.2, 0.9, 0.3, 0.4],    # 正确类别: 2
                [0.8, 0.1, 0.2, 0.3, 0.4],    # 正确类别: 0
                [0.1, 0.9, 0.2, 0.3, 0.4],    # 正确类别: 1
                [0.1, 0.2, 0.3, 0.4, 0.9]]    # 正确类别: 4
    predict("model_multi_class.bin", test_vec)

