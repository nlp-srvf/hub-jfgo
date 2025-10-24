import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())
        else:
            return torch.softmax(y_pred, dim=-1)

# 生成样本：最大值所在的索引决定类别
def build_sample():
    x = np.random.random(5)
    y = np.argmax(x)  # 最大值所在的索引作为类别标签
    return x, y

# 生成数据集
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 评估函数
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    # 统计各类别样本数量
    class_counts = [0] * 5
    for label in y:
        class_counts[int(label)] += 1
    print("各类别样本数量:", class_counts)
    correct = 0
    with torch.no_grad():
        y_prob = model(x)  # 获取概率分布
        y_pred = torch.argmax(y_prob, dim=1)  # 取概率最大的类别
        correct = torch.sum(y_pred == y.squeeze()).item()
    acc = correct / test_sample_num
    print(f"正确预测个数: {correct}, 正确率: {acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 20 # 训练轮数
    batch_size = 20 # 每次训练样本个数
    train_sample = 5000 # 每轮训练总共训练的样本总数
    input_size = 5 # 输入5维
    num_classes = 5  # 输出5维
    learning_rate = 0.001 # 学习率
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
        # 打乱数据
        indices = torch.randperm(train_sample)
        train_x = train_x[indices]
        train_y = train_y[indices]
        for i in range(0, train_sample, batch_size):
            # 取出一个batch数据作为输入
            x_batch = train_x[i:i + batch_size]
            y_batch = train_y[i:i + batch_size]
            loss = model(x_batch, y_batch) # 计算loss  model.forward(x,y)
            loss.backward() # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad() # 梯度归零
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, avg_loss))
        acc = evaluate(model) # 测试本轮模型结果
        log.append([acc, avg_loss])
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    model = TorchModel(5, 5)
    model.load_state_dict(torch.load(model_path)) # 加载训练好的权重
    print(model.state_dict())
    model.eval() # 测试模式
    with torch.no_grad(): # 不计算梯度
        result = model(torch.FloatTensor(input_vec)) # 模型预测
        predictions = torch.argmax(result, dim=1)
    for vec, prob, pred in zip(input_vec, result, predictions):
        print(f"输入: {vec}")
        print(f"类别概率: {[f'{p:.4f}' for p in prob]}")
        print(f"预测类别: {pred.item()}\n")


if __name__ == "__main__":
    main()
    # # 测试预测
    # test_vectors = [
    #     [0.9, 0.1, 0.2, 0.3, 0.4],  # 第0类
    #     [0.1, 0.95, 0.2, 0.3, 0.4],  # 第1类
    #     [0.1, 0.2, 0.98, 0.3, 0.4],  # 第2类
    #     [0.1, 0.2, 0.3, 0.99, 0.4],  # 第3类
    #     [0.1, 0.2, 0.3, 0.4, 0.97]  # 第4类
    # ]
    # predict("model.bin", test_vectors)
