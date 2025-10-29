import torch
import torch.nn as nn
import numpy as np
import random
import json

def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index, ch in enumerate(chars):
        vocab[ch] = index+1
    vocab['unk'] = len(vocab)
    return vocab

def build_sample(vocab, sentence_length):
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    x.append('pad')
    if '你' in x:
        y = x.index("你")
    else:
        y = len(x) - 1
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y

def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab, char_dim, sentence_length, hidden_size):
    model = TorchModel(char_dim, sentence_length, hidden_size, vocab)
    return model

def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)
    print("本次预测集中共有%d个正样本，%d个负样本"%(sum(y), 200 - sum(y)))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred =model(x)
        for y_p, y_t in zip(y_pred, y):
            if y_p.argmax().item() == y_t:
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))
    return correct/(correct+wrong)

def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6 + 1  # 样本文本长度, 加上一个pad字符
    hidden_size = 16  # 隐藏层大小
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length, hidden_size)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        py_x = [vocab[char] for char in input_string]
        py_x.append(vocab['pad'])
        x.append(py_x)  #将输入序列化
    x = torch.LongTensor(x)
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        result = model.forward(x)  #模型预测
    for i, input_string in enumerate(input_strings):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, result[i].argmax().item(), result[i].max().item())) #打印结果

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, hidden_size, vocab):
        super(TorchModel, self).__init__()
        self.embeddidng = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        self.pool = nn.AvgPool1d(hidden_size)
        self.rnn_layer = nn.RNN(vector_dim, hidden_size, batch_first=True)
        self.activation = torch.nn.Softmax(dim=1)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y=None):
        x = self.embeddidng(x) # [batch_size, sentence_length, vector_dim]
        y_pred, _ = self.rnn_layer(x) # [batch_size, sentence_length, hidden_size]
        y_pred = self.pool(x).squeeze(-1) # [batch_size, sentence_length]
        # y_pred = self.activation(x)
        if y is not None:
            return self.loss(y_pred, y)
        return self.activation(y_pred)

def main():
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6 + 1   #样本文本长度, 加上一个pad字符
    hidden_size = 16  # 隐藏层大小
    learning_rate = 0.005 #学习率

    vocab = build_vocab()
    model = build_model(vocab, char_dim, sentence_length, hidden_size)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    torch.save(model.state_dict(), "model.pth")
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    
    

if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
