import os
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn


class MyModel(nn.Module):
    '''
    自定义模型 - 五分类任务
    '''
    def __init__(self, input_size, num_classes=5):
        super(MyModel, self).__init__()
        # 线性层：输入是5维，输出是5个类别
        self.linear = nn.Linear(input_size, num_classes)
        # 使用交叉熵损失函数，适用于多分类任务
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        '''
        前向计算
        :param x: 输入一个五维的向量
        :param y: 真实标签（类别索引）
        :return: 预测结果或损失值
        '''
        # 线性层：(batch_size, input_size) --> (batch_size, num_classes)
        logits = self.linear(x)
        if y is not None:
            # 如果传入了标签y，计算交叉熵损失
            # y应该是LongTensor类型，包含类别索引（0-4）
            return self.loss(logits, y.long())
        else:
            # 如果没有传入标签y，返回原始logits
            return logits


def get_dataset(train_sample):
    """
    获取训练集，随机生成训练数据
    :param train_sample: 训练样本数量
    :return: 训练集 {train_x, train_y}
    """
    X = []
    Y = []
    for i in range(train_sample):
        # 生成一个五维向量
        x = np.random.random(5)
        # 标签为最大值索引（0-4）
        y = np.argmax(x)  # 返回最大值的索引位置
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 注意标签需要是LongTensor


def evaluate(model):
    """
    模型效果测试方法
    :param model: 模型实例
    :return: 准确率
    """
    model.eval()
    test_sample_num = 100
    x, y = get_dataset(test_sample_num)
    # 统计每个类别的样本数量
    class_counts = [torch.sum(y == i).item() for i in range(5)]
    print("本次预测集中各类别样本数量:", class_counts)
    correct, wrong = 0, 0
    with torch.no_grad():
        logits = model(x)  # 获取模型输出的logits
        predictions = torch.argmax(logits, dim=1)  # 获取预测类别
        correct = (predictions == y).sum().item()
        wrong = test_sample_num - correct
    accuracy = correct / (correct + wrong)
    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    """
    主训练函数
    """
    # 配置参数
    epoch_num = 500  # 训练轮数
    batch_size = 20  # 批次大小
    train_sample = 5000  # 每轮训练样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 类别数量
    learn_rate = 0.01  # 学习率
    # 建立模型
    model = MyModel(input_size, num_classes)
    # 选择优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    log = []
    # 读取训练集
    train_x, train_y = get_dataset(train_sample)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            # 获取当前batch数据
            start_idx = batch_index * batch_size
            end_idx = (batch_index + 1) * batch_size
            x = train_x[start_idx:end_idx]
            y = train_y[start_idx:end_idx]
            # 计算损失
            loss = model(x, y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            watch_loss.append(loss.item())
        # 每10轮打印一次训练信息
        if epoch % 10 == 0:
            avg_loss = np.mean(watch_loss)
            print("====> 第[%d]轮训练，平均loss：%f" % (epoch + 1, avg_loss))
            acc = evaluate(model)
            log.append([acc, avg_loss])
    # 保存模型
    os.makedirs("tmp/model", exist_ok=True)
    torch.save(model.state_dict(), "tmp/model/MyModel02.pt")
    # 绘制训练曲线
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(len(log)), [l[0] for l in log], label="Accuracy")
    # plt.xlabel("Epoch")
    # plt.ylabel("Accuracy")
    # plt.legend()
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(len(log)), [l[1] for l in log], label="Loss", color='red')
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.savefig("tmp/model/training_curve.png")
    # plt.show()
    return


def predict(model_path, input_vec):
    """
    模型预测函数
    :param model_path: 模型路径
    :param input_vec: 需要预测的输入向量列表
    """
    input_size = 5
    num_classes = 5
    model = MyModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    with torch.no_grad():
        # 将输入转换为张量
        input_tensor = torch.FloatTensor(input_vec)
        # 模型预测
        logits = model(input_tensor)
        # 获取预测概率（使用softmax）
        probabilities = torch.softmax(logits, dim=1)
        # 获取预测类别
        predictions = torch.argmax(logits, dim=1)
        print(f"模型权重为：{model.state_dict()}")
        print("====================> 模型预测结果 <====================")
        for i, (vec, pred, prob) in enumerate(zip(input_vec, predictions, probabilities)):
            actual_class = np.argmax(vec)  # 实际类别（最大值索引）
            is_correct = "✓" if pred.item() == actual_class else "✗"
            print(f"输入{i + 1}: {[f'{v:.3f}' for v in vec]} === 类别{pred.item()} {is_correct} === 实际类别: {actual_class}")


if __name__ == '__main__':
    # 训练模型
    main()
    # 测试预测
    test_vec = [
        [0.9, 0.15, 0.31, 0.04, 0.85],  # 第1个最大 → 类别0
        [0.75, 0.55, 0.96, 0.96, 0.85],  # 第3或4个最大 → 类别2或3
        [0.67, 0.14, 0.35, 0.20, 0.91],  # 第5个最大 → 类别4
        [0.99, 0.59, 0.93, 0.42, 0.14]  # 第1个最大 → 类别0
    ]
    predict("tmp/model/MyModel02.pt", test_vec)
