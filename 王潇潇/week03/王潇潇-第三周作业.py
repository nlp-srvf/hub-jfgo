# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的单向RNN网络
任务：判断特定字符（你/我/他）在文本中的位置（0-5，共6个位置）
     若不存在则输出6
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, num_classes=7):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)

        # 定义单向RNN层
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=vector_dim,
            num_layers=1,
            bidirectional=False,
            batch_first=True
        )
        # 线性层
        self.classify = nn.Linear(vector_dim, num_classes)
        self.loss = nn.CrossEntropyLoss() # 多分类任务需要使用交叉熵

    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sentence_length) -> (batch_size, sentence_length, vector_dim)
        rnn_out, _ = self.rnn(x)  # (batch_size, sentence_length, vector_dim)
        logits = self.classify(rnn_out)  # (batch_size, sentence_length, num_classes)

        if y is not None:
            # 不存在就返回最后一个向量
            return self.loss(logits.reshape(-1, 7), y.reshape(-1))
        else:
            return torch.softmax(logits, dim=-1)


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}  # 填充字符用单个"pad"表示
    for index, char in enumerate(chars):
        vocab[char] = index + 1
    vocab['unk'] = len(vocab) # 不存在的字符使用“unk”表示
    return vocab


def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    y = [6] * sentence_length  # 初始化标签为无特定字符
    for i in range(sentence_length):
        if x[i] in {"你", "我", "他"}:
            y[i] = i  # 标记特定字符位置
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 初始化数据
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    # 多分类任务
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 创建模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x) # 模型预测
        # 因为是多分类任务 所以需要识别概率的索引位置
        y_pred_label = torch.argmax(y_pred, dim=-1)
        for b in range(len(x)):
            for i in range(sentence_length):
                if y_pred_label[b][i] == y[b][i]: # 和真实值做对比
                    correct += 1
                else:
                    wrong += 1
    acc = correct / (correct + wrong)
    print(f"正确预测个数：{correct}, 错误预测个数：{wrong}, 正确率：{acc:.4f}")
    return acc

# 训练模型
def main():
    epoch_num = 20
    batch_size = 20
    train_sample = 1000
    char_dim = 20
    sentence_length = 6
    learning_rate = 0.001

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())

        print(f"=========\n第{epoch + 1}轮平均loss: {np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "homework.pth")
    with open("vocab.json", "w", encoding="utf8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)
    return

# 测试
def predict(model_path, vocab_path, input_strings):
    char_dim = 20
    sentence_length = 6
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    x = []
    for input_string in input_strings:
        # 使用单个字符"_"作为填充符，而不是"pad"
        if len(input_string) > sentence_length:
            input_str = input_string[:sentence_length]
        else:
            # 用单个填充字符补全，而不是多字符字符串
            input_str = input_string.ljust(sentence_length, "_")  # 单字符填充
        x.append([vocab.get(char, vocab['unk']) for char in input_str])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        pred_label = torch.argmax(result, dim=-1)

    for i, input_string in enumerate(input_strings):
        print(f"输入：{input_string}")
        print("位置预测：", end="")
        isExit=False
        for pos in range(sentence_length):
            # 判断位置 如果位置大于长度 说明这个字符没有出现过
            char = input_string[pos] if pos < len(input_string) else "_"
            pred = pred_label[i][pos].item()
            if(pred == pos):
                # 判断是位置预测的判断位置
                isExit=True
                print(f"第{pos}位[{char}]是特定字符")
        if not isExit:
            print("该字符不存在特定字符！")
        print("=====")

if __name__ == "__main__":
    main()
    test_strings = ["我abcde", "a你cdef", "abcde他", "abcdef", "你我他abc"]
    predict("homework.pth", "vocab.json", test_strings)
