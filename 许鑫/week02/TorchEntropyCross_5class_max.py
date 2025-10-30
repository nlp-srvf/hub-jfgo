import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')
# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=5, out_features=5)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def build_samples():
    x = np.random.random(5)
    # y = np.zeros(5)
    # 5维随机向量最大位置为1标签
    # y[np.argmax(x)] = 1
    # nn.CrossEntropyLoss() 期望的是 类别索引
    y = np.argmax(x)

    return x, y


def build_dataset(total_sample_num):
    X, Y = [], []
    for i in range(total_sample_num):
        x, y = build_samples()
        X.append(x)
        Y.append(y)
    X = np.array(X, dtype='float32')
    Y = np.array(Y, dtype='int64')
    return torch.from_numpy(X), torch.from_numpy(Y)


def evaluate(model):
    model.eval()
    test_sample_num = 1000
    x, y = build_dataset(test_sample_num)
    # print(x, '/n', y)
    correct, wrong = 0, 0
    with torch.no_grad():
        x = x.to(device)
        y = y.to(device)

        y_pred = model(x)
        # input(y_pred)
        max_index = torch.argmax(y_pred, dim=1)
        # input(max_index)
        correct += (max_index == y).sum().item()
        unique, counts = torch.unique(y, return_counts=True)
        for cls, cnt in zip(unique.tolist(), counts.tolist()):
            cls_correct = (max_index[y == cls] == cls).sum().item()
            print(f'类别 {cls}: {cls_correct}/{cnt} ({cls_correct / cnt:.4f})')

    # print(f'正确预测个数{correct}, 正确率为{correct / test_sample_num}')
    # input()
    return correct / test_sample_num


def main():
    epoches = 200
    batch_size = 2000
    learning_rate = 1e-2
    train_sample_num = 500000

    model = TorchModel()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    train_x, train_y = build_dataset(train_sample_num)
    for epoch in range(epoches):
        model.train()
        watch_loss = []
        for i in range(train_sample_num // batch_size):
            x = train_x[i * batch_size: (i + 1) * batch_size]
            y = train_y[i * batch_size: (i + 1) * batch_size]
            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optim.step()
            optim.zero_grad()
            watch_loss.append(loss.item())

        print(f'第{epoch}轮平均loss为: {np.mean(watch_loss):.4f}')
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    # torch.save(model.state_dict(), './model.pt')
    plt.plot(range(len(log)), [l[0] for l in log], label='acc')
    plt.plot(range(len(log)), [l[1] for l in log], label='loss')
    plt.legend()
    plt.savefig('acc_loss.png')
    return


if __name__ == '__main__':
    main()
