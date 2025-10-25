import torch
import torch.nn as nn
import numpy as np
"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪一维的数最大则分到哪一类，比如第二个数最大则分到第二类，诸如此类

"""

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)  # 输出logits（无激活）
        self.loss = nn.functional.cross_entropy  # 使用交叉熵损失函数会默认对每一行进行softmax归一化，因此不需要提前进行softmax归一化

    def forward(self, x, y=None):
        y_pred = self.linear(x)  # 直接输出logits，不做激活
        if y is not None:
            return self.loss(x, y)  # y是类别索引
        else:
            return y_pred

# 生成样本：标签为类别索引（整数），而非one-hot
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 最大值下标（0-4），直接作为标签
    return x, max_index

# 生成数据集：标签为LongTensor（整数类型）
def build_dataset(total_sample_num):
    X = []
    Y = []
    for _ in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X), torch.LongTensor(Y)  # 标签用LongTensor

def main():
    epoch_num = 1000
    batch_size = 20
    train_sample = 10000
    input_size = 5
    output_size = 5
    lr = 0.1  # 学习率这里设置为0.1收敛会快一点，一开始使用0.001测试，收敛很慢，故改为0.1测试，收敛速度适中

    model = TorchModel(input_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_x, train_y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_idx in range(train_sample // batch_size):
            x = train_x[batch_idx*batch_size : (batch_idx+1)*batch_size]
            y = train_y[batch_idx*batch_size : (batch_idx+1)*batch_size]
            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print(f"第{epoch+1}轮，平均loss：{np.mean(watch_loss):.6f}")

    torch.save(model.state_dict(), "fiveclassmodel.bin")
#最后通过argmax取最大值索引即可看到预测效果
#老师看到这份作业的话麻烦看下思路有没有问题，有问题烦请反馈给我哈

if __name__ == "__main__":
    main()
