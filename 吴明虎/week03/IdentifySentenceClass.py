import os
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import random
import json
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 作业，使用5分类，判断给与的词A在文本中哪个位置就算哪类

class IdentifySentenceClass(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab, hidden_size=10):
        super(IdentifySentenceClass, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)
        # self.pool = nn.AvgPool1d(sentence_length)
        self.identity = nn.RNN(sentence_length, hidden_size, bias=False, batch_first=True)
        self.loss = nn.CrossEntropyLoss()



    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x = x.transpose(1, 2)                      #(batch_size, sen_len, vector_dim) -> (batch_size, vector_dim, sen_len)
        # x = self.pool(x)                         # pooling 后不收敛
        # x = x.squeeze()
        output,h = self.identity(x)                #(batch_size, vector_dim, sen_len) -> (batch_size, vector_dim, vector_dim) +  (1, batch_size, vector_dim)
        # print(output.detach().numpy(), "torch模型预测结果",output.shape)
        # print(h.detach().numpy(), "torch模型预测隐含层结果",h.shape)
        y_pred=h.squeeze()
        if y is not None:
            return self.loss(y_pred, y)  #预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1) #输出预测结果


def build_vocab():
    charset = "你我他defghijklmnopqrstuvwxyz"
    vocab = {"pad": 0}
    for idx, char in enumerate(charset):
        vocab[char] = idx + 1
    vocab["UNK"] = len(charset)
    return vocab

def random_data(word,target):
    p =  np.random.randint(0, 7)
    if 5 > p > -1:
        word[p] = target
    return word

def build_sample(vocab, sentence_length):
    target = "我"
    #随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    #出现指定字在第几行就算第几类，但上列列表可能不存在指定字# 可能没有指定字，但不能总是没有 #设计一个随机函数修改
    #出现多个目标值就只取第一个
    y=0
    x = random_data(x, target)
    if target in x:
        y=x.index(target)+1
    #指定字都未出现，则为0样本
    # x = [vocab.get(word, vocab['UNK']) for word in x]  #将字转换成序号，为了做embedding
    x = [vocab.get(word) for word in x]  # 将字转换成序号，为了做embedding
    return x, y


#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    X = []
    Y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        X.append(x)
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y)


#建立模型
def build_model(char_dim, sentence_length, vocab):
    model = IdentifySentenceClass(char_dim, sentence_length, vocab, 10)
    return model


#评估模型输出结果与标签对应率
def evaluate(model, vocab,sentence_length):
    model.eval()
    test_sample_num = 200
    x, y = build_dataset(test_sample_num,vocab, sentence_length)

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
            if torch.argmax(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)



def main():
    # 配置参数
    epoch_num = 100  # 训练轮数
    batch_size = 200  # 每次训练样本个数
    train_sample = 6000  # 每轮训练总共训练的样本总数
    char_dim = 10  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    learning_rate = 0.005  # 学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(char_dim, sentence_length, vocab)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            #print(loss)
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    print(log)
    # 展示结果
    plt.plot(range(len(log)), [a[0] for a in log], label="acc")
    plt.plot(range(len(log)), [a[1] for a in log], label="loss")
    plt.legend()
    plt.show()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 10  # 每个字的维度
    sentence_length = 5  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(char_dim, sentence_length, vocab)   #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    x = []
    for input_string in input_strings:
        input_string = input_string[:5]   #多余字符串删除
        xi=[]
        for char in input_string:
            if char in vocab:
                xi.append(vocab[char])  #将输入序列化
            else:
                xi.append(vocab["UNK"])
        x.append(xi)
    # print(x,len(x))
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.LongTensor(x))  # 模型预测
    for ins, res in zip(input_strings, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (ins, torch.argmax(res),res)) #打印结果


if __name__ == '__main__':
    main()
    test_strings = ["fnf我e", "wz你dg", "rqdeg", "n我kww", "n我kww" , "wo我aq" ,"hucef" ]  #a/b/c不存在字典
    predict("model.pth", "vocab.json", test_strings)
  
