# 5维向量，判断哪一个维度的值最大，就是第几类
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# 1.建立线性层、采用交叉熵损失函数（选择模型结构）
class TorchModel2(nn.Module):
    def __init__(self,input_size):
        super(TorchModel2, self).__init__()
        self.linear = nn.Linear(input_size,5) #线性层
        self.loss = nn.CrossEntropyLoss() #loss函数采用交叉熵损失函数（自带softmax归一化）

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            y_classes = torch.argmax(y, dim=1)
            return self.loss(x,y_classes)  #预测值
        else:
            return x

# 2.生成一个样本，5维向量（训练样本选择）
def bulid_sample():
    x = np.random.random(5)
    m = np.argmax(x)
    y = [0] * 5
    y[m] = 1
    return x,y,m

# 3.随机生成一批样本（训练样本生成）
def build_dataset(total_sample_num):
    X = []
    Y = []
    Z = [0] * 5
    for i in range(total_sample_num):
        x, y,m = bulid_sample()
        X.append(x)
        Y.append(y)
        Z[m] += 1
    return torch.FloatTensor(X), torch.FloatTensor(Y),Z

# 4.测试代码准确率
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 200
    x,y,z = build_dataset(test_sample_num)
    print("这次预测了%d个一型样本，%d个二型样本，%d个三型样本，%d个四型样本，%d个五型样本" % (z[0], z[1], z[2], z[3], z[4]))

    correct,wrong = 0,0
    with torch.no_grad():
        y_pred = model(x)
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        for y_p,y_t in zip(y_pred_softmax, y):
            if torch.argmax(y_p) == torch.argmax(torch.tensor(y_t)):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率%f" %(correct,correct / (correct + wrong)))
    return correct / (correct + wrong)

# 5.选择优化器、训练模型
def main():
    # 配置参数
    epoch_num = 800 #训练轮数
    batch_size = 100 #每次训练样本个数
    train_sample = 5000 #每轮训练总共训练的样本总数
    input_size = 5
    learning_rate = 0.0001

    #建立模型
    model = TorchModel2(input_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y, train_z = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size:(batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size:(batch_index + 1) * batch_size]

            loss = model(x,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())

        print("================\n第%d轮平均loss:%f" %(epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log],label="acc")
    plt.plot(range(len(log)), [l[1] for l in log],label="loss")
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    main()



