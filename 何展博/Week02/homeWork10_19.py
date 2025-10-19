import torch
import torch.nn as nn
import numpy as np
import random
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, 5)  # 输出5个类别
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失
    # 正向传播
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss(x, y)
        else:
            return x


def build_sample():
    x = np.random.random(5)  # 生成5维随机向量
    label = np.argmax(x)  # 获取最大值的索引
    return x, label

# 构建测试数据
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)  # 直接存储索引

    # 转换为Tensor
    return torch.FloatTensor(X), torch.LongTensor(Y)


def evaluate(model):
    """评估模型性能"""
    model.eval()
    test_sample_num = 2
    x, y = build_dataset(test_sample_num)
    print('x',x)
    print('y',y)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)
        # 获取预测的类别 (argmax)
        pred = y_pred.argmax(dim=1)

        # 比较预测和真实标签
        for i in range(len(pred)):
            if pred[i] == y[i]:
                correct += 1
            else:
                wrong += 1

    accuracy = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 正确率：{accuracy:.4f}")
    return accuracy


def main():
    epoch_num = 20  # 增加训练轮数
    batch_size = 20
    train_sample = 10000
    input_size = 5
    learning_rate = 0.1  # 提高学习率

    model = TorchModel(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            start = batch_index * batch_size
            end = start + batch_size
            x = train_x[start:end]
            y = train_y[start:end]

            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.4f}")
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model.bin')
    print(f"训练完成，最终准确率: {log[-1][0]:.4f}")

    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(log)), [l[0] for l in log], 'b-', label="Accuracy")
    plt.plot(range(len(log)), [l[1] for l in log], 'r-', label="Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()


def predict(model_path, input_vec):
    """预测函数"""
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        result = model(torch.FloatTensor(input_vec))
        # 预测最大索引 (argmax)
        pred = result.argmax(dim=1)

        # 打印预测结果
        for i, vec in enumerate(input_vec):
            print(f"输入：{vec}, 预测最大索引：{pred[i].item()}, 概率分布：{result[i].detach().numpy()}")

    return pred


if __name__ == '__main__':
     main()
    #测试预测
    # test_vec = [
    #     [0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #     [0.74963533, 0.5524256, 0.95758807, 0.95520434, 0.84890681],
    #     [0.00797868, 0.67482528, 0.13625847, 0.34675372, 0.19871392],
    #     [0.09349776, 0.59416669, 0.92579291, 0.41567412, 0.1358894]
    # ]
    # predict("model.bin", test_vec)