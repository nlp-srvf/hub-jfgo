import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# 用交叉熵实现一个多分类任务，五维随机向量最大的数字在哪维就属于哪一类。


# 选择模型结构,线性层+交叉熵损失函数
class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(TorchModel, self).__init__()
        #定义要使用哪些网络层
        self.linear = nn.Linear(input_size, num_classes)
        self.loss = nn.CrossEntropyLoss()

    #y是标签，输入标签时返回loss，没有标签就返回预测值
    def forward(self, x, y=None):
        y_pred = self.linear(x)
        if y is not None:
            y_class=torch.argmax(y,dim=1)
            return self.loss(y_pred, y_class)
        else:
            return y_pred

# 随机生成给与维度的向量，
def build_sample(data_dim):
    x=np.random.random(data_dim)
    a=np.argmax(x)
    y=np.zeros(data_dim)
    y[a]=1
    return x,y

# 生成模拟训练样本
def build_dataset(sample_nums,data_dim):
    X=[]
    Y=[]
    for i in range(sample_nums):
        x,y=build_sample(data_dim)
        X.append(x)
        Y.append(y)
    return torch.FloatTensor(X),torch.FloatTensor(Y)

#评估模型输出结果与标签对应率
def evaluate(model,input_size):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num,input_size)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测 model.forward(x)
        y_pred_softmax = torch.softmax(y_pred, dim=1)
        for y_p, y_t in zip(y_pred_softmax, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == torch.argmax(torch.tensor(y_t)):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

# 主函数，获得数据集，训练并优化模型
def main():
    # 初始化参数
    epoch_num = 200  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 10000  # 每轮训练总共训练的样本总数
    input_size = 5
    learning_rate = 0.001

    num_classes = 5
    model = TorchModel(input_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y= build_dataset(train_sample,input_size)
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []

        for batch in range(train_sample // batch_size):
            x = train_x[batch*batch_size:(batch+1)*batch_size]
            y = train_y[batch*batch_size:(batch+1)*batch_size]

            loss = model(x, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        loss_val = np.mean(watch_loss)
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, loss_val))
        acc = evaluate(model, input_size)
        log.append([acc, float(loss_val)])

    # 保存模型
    torch.save(model.state_dict(), "valuue_class_model.bin")
    print(log)
    # 展示结果
    plt.plot(range(len(log)), [a[0] for a in log], label="acc")
    plt.plot(range(len(log)), [a[1] for a in log], label="loss")
    plt.legend()
    plt.show()


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size,num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d" % (vec, torch.argmax(res)+1))  # 打印结果


if __name__ == '__main__':
    main()
    # test_vec = [[0.07889086, 0.15229675, 0.31082123, 0.03504317, 0.88920843],
    #             [0.74963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("valuue_class_model.bin", test_vec)
    
