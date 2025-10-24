import torch.nn as nn
import numpy as np
import torch
import random
import matplotlib.pyplot as plt


def build_sample(nums):
    X = np.random.random((nums, 5))
    return X

def one_hot_encode(y):
    Y = np.zeros((y.shape[0], 5))
    for i in range(y.shape[0]):
        Y[i][y[i]] = 1
    return Y

def build_dataset(nums):
    X = build_sample(nums)
    Y = np.argmax(X, axis=1)
    Y = one_hot_encode(Y)

    return torch.FloatTensor(X), torch.FloatTensor(Y)


class Torchmodel(nn.Module):
    def __init__(self, input_size):
        super(Torchmodel, self).__init__()
        self.linear = nn.Linear(input_size, 5)
        self.softmax = nn.functional.softmax
        self.loss = nn.functional.cross_entropy

    def forward(self, X, Y= None):
        X = self.linear(X)
        Y_pred = self.softmax(X, dim=1)
        if Y is not None:
            return self.loss(Y_pred, Y)
        else:
            return Y_pred


def evaluate(model):
    model.eval()
    test_sample_nums = 100
    X, Y = build_dataset(test_sample_nums)
    print(f"本次测试一共有{test_sample_nums}个样本")
    correct, wrong = 0, 0
    with torch.no_grad():
        Y_pred = model(X)
        for y_p, y_t in zip(Y_pred, Y):
            if torch.argmax(y_p) == torch.argmax(y_t):
                correct += 1
            elif torch.argmax(y_p) != torch.argmax(y_t):
                wrong += 1

    print(f"本次测试正确预测个数为：{correct}, 正确率为：{correct/test_sample_nums:.2f}")
    return correct / test_sample_nums


def main():
    epoch_num = 10
    batch_size = 20
    train_sample = 5000
    input_size = 5
    lr = 0.001

    model = Torchmodel(input_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log = []

    train_X, train_Y = build_dataset(train_sample)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            X = train_X[batch_index * batch_size:(batch_index + 1) * batch_size]
            Y = train_Y[batch_index * batch_size:(batch_index + 1) * batch_size]
            loss = model(X, Y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())
            print(f"<==============\n第{epoch+1}轮平均loss:{np.mean(watch_loss):.3f}")
            acc = evaluate(model)
            log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), 'model.bin')

    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    return None


def predict(model_path, input_vec):
    input_size = 5
    model = Torchmodel(input_size)
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.FloatTensor(input_vec))
        for vec, res in zip(input_vec, result):
            print(f"输入：{input_vec}, 预测结果：{result}")

if __name__ == '__main__':
    main()
