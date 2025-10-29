import torch
import torch.nn as nn
import numpy as np

"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
任务要求：完成一个5分类任务，输入5维向量，判断其中最大数字在哪一维就是属于哪一类
"""


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 定义线性层，输入5维，输入出5维
        self.loss = nn.CrossEntropyLoss()  # 定义loss使用交叉熵

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值与真实值计算loss
        else:
            return y_pred


# x = 随机生成5维向量， y = 5维向量，x中最大值的维度维1，其它维度维0
# x = [1,2,3,4,5]   y = [0,0,0,0,1]
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(np.array(x))
    y = [1 if i == max_index else 0 for i in range(5)]
    return x, y


# 构建训练样本数据集
def build_dataset(data_total):
    x_arr = []
    y_arr = []
    for i in range(data_total):
        x, y = build_sample()
        x_arr.append(x)
        y_arr.append(y)
    return torch.FloatTensor(x_arr), torch.FloatTensor(y_arr)


# 测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_data_total = 100
    x, y = build_dataset(test_data_total)
    c, w = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
            if np.argmax(y_p) == np.argmax(y_t):
                c += 1
            else:
                w += 1
    print("测试样本数：{}，正确数：{}，正确率：{}".format(test_data_total, c, c / (c + w)))
    return c / (c + w)


def main():
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本数
    train_sample = 5000  # 每轮训练总共训练样本总数
    input_size = 5  # 输入向量维度
    lr = 0.001  # 学习率

    # 建立模型
    model = TorchModel(input_size)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # 创建训练样本
    x_train, y_train = build_dataset(train_sample)
    # 训练过程
    for i in range(epoch_num):
        model.train()
        loss_arr = []
        for batch_index in range(train_sample // batch_size):
            x = x_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            y = y_train[batch_index * batch_size: (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            loss_arr.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (i + 1, np.mean(loss_arr)))
        evaluate(model)  # 测试本轮模型结果
    torch.save(model.state_dict(), 'model.pt')
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec,  round(float(torch.argmax(res)))))  # 打印结果


if __name__ == "__main__":
    # main()
    test_vec = [[1,2,3,4,5],
                [6,2,3,5,1],
                [2,31,4,2,9],
                [11,101,33,22,99],
                [21.75,3.45,41,2,9],
                [112,14,4,2,9],
                [2.9,131,4.78,2,9]]
    predict("model.pt", test_vec)
