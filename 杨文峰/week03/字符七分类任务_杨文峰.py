import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
尝试在nlpdemo中使用rnn模型训练，判断特定字符在文本中的位置，五个字符。
"""

class TorchModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim,num_classes=7):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)  #词嵌入层，词汇表大小+向量维度+0填充
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)  #双向长短期记忆网络，特征维度+隐藏维度+输出张量确定为batch_size、seq_len、features（样本数量，序列长度，特征向量维度）、双向处理
        self.dropout = nn.Dropout(0.3)   #选择丢弃30%数据，防止过拟合
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)   #在LSTM后接收hidden层*2的特征，转换为7分类任务
        self.loss = nn.CrossEntropyLoss()  #交叉熵损失

    def forward(self, x, y=None):   #定义前向传播，接收输入数据x和可选标签y，如果没y则默认为None
        x = self.embedding(x)       #将字符串通过embedding层转换成密集的向量
        rnn_out, (hidden, _) = self.rnn(x)    #将嵌入的序列输入RNN层进行处理，得到每一个时间步的输出和最后一个时间步的隐藏状态（理解为rnn_out的最后一个输出、总结）
        x = torch.cat((hidden[-2], hidden[-1]), dim=-1) #cat拼接最后两个时间步的隐藏状态，分别获取前向和后向的最终隐藏状态
        x = self.dropout(x)   #正则化，随机丢弃部分神经元输出、防止模型过拟合、提高泛化能力
        x = self.classifier(x) #转化为7分类任务
        if y is not None:
            return self.loss(x, y)  #训练的话直接计算损失函数
        else:
            return torch.softmax(x, dim=-1)  #预测的话返回各类别的概率分布

def build_vocab():   #构建一个字符串的字典，0填充，其余一个对应一个数字，未出现的字符用最后一位数字填充
    chars = "你我他abcdefghijklmnopqrstuvwxyz"
    vocab = {"pad":0}
    for index,char in enumerate(chars):
        vocab[char] = index+1
    vocab["unk"] = len(vocab)
    return vocab

def build_sample(vocab, sentence_len):  #随机选择一个字典中的值（先转成列表）填充进x，长度为sentence_len，计算‘你’出现的次数以及位置
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_len)]
    count = x.count('你')
    if count == 0:
        y = 0
    elif count == 1:
        for i, char in enumerate(x):
            if char == '你':
                y = i + 1
                break
    else:
        y = 6

    x = [vocab.get(char, vocab["unk"]) for char in x] #遍历x中的每个字符char，看输入的x字符串是否在字典中，如果在则返回对应索引，不在返回unk的索引
    return x, y


def build_dataset(sample_len,vocab, sentence_len): #建立数据集的序列，转换为整数型张量
    dataset_x = []
    dataset_y = []
    for i in range(sample_len):
        x, y = build_sample(vocab, sentence_len)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model(vocab_size,embedding_dim, hidden_dim,num_classes=7): #封装了TorchModel的实例化过程，简化模型创建
    model = TorchModel(vocab_size, embedding_dim, hidden_dim,num_classes)
    return model

def evaluate(model,vocab,sample_len,num_classes=7):  #测试模型
    model.eval() #进入评估模式，停用Dropout层（与model.train()对应，这是进入训练模式）
    x,y_true = build_dataset(200,vocab,sample_len)

    with torch.no_grad():  #无梯度环境节省内存并提高计算效率
        outputs = model(x)  #将输入数据x传入模型获得原始输出结果
        y_pred = torch.softmax(outputs, dim=1)  #对模型的输出应用softmax函数归一化
        _, predicted_classes = torch.max(y_pred, 1)  #获取每个样本预测概率最大的类别索引，_表示忽略最大值本身
        correct = (predicted_classes == y_true).sum().item()  #通过预测类别和真实标签来确认正确预测样本数
        total = len(y_true) #总样本数
        accuracy = correct / total #正确率

        counts = {}
        for i in range(num_classes):
            counts[i] = (y_true == i).sum().item() #每个类别的数量

    print("此次预测结果:")
    for i in range(num_classes):
        print(f"  类别{i}: {counts[i]}个样本")
    print(f"  正确预测: {correct}个，错误预测: {total - correct}个，准确率: {accuracy:.4f}")
    return accuracy

def main():
    epoch_num = 10  #训练轮数
    batch_size = 32  #每批训练大小
    train_samples = 500  #训练样本数
    embedding_dim = 32  #词向量维度
    hidden_dim = 128  #隐藏层维度
    sentence_len = 5  #句长
    learning_rate = 0.005  #学习率

    vocab = build_vocab() #建立词汇字典
    model = build_model(len(vocab),embedding_dim,hidden_dim,num_classes=7)  #使用模型，代入各参数，词汇表大小为字典内长度

    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)  #引入Adam优化器
    log = []  #建立一个数列储存
    for epoch in range(epoch_num):
        model.train()  #训练模式
        watch_loss = []  #每次训练用数列存储训练过程中所有损失值
        for batch in range(int(train_samples/batch_size)):  #训练样本数除以每批训练大小得到间隔数
            x, y = build_dataset(train_samples,vocab,sentence_len)
            optim.zero_grad() #每次使用前梯度需归零，不然会累加上之前的梯度
            loss = model(x, y)  #计算与真实值的损失函数
            loss.backward()  #计算梯度
            optim.step()  #更新优化器的参数

            watch_loss.append(loss.item()) #记录损失值
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_len)
        log.append([acc,np.mean(watch_loss)])

    torch.save(model.state_dict(), "model2.pth")  #model.state_dict()可以获得模型的权重和偏置，然后将模型参数保存到文件model2.pth
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")  #以写入模式打开vocab.json，指定UTF-8编码确保中文字符正确保存
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2)) #将词汇表对象转换为json格式字符串，ensure_ascii关闭允许保存非ASCII字符（如中文），indent=2使格式美观，每个层级缩进2个空格，数值越大层级越清晰
    writer.close() #关闭文件句柄，确认数据写入磁盘并释放系统资源
    return

def predict(model_path,vocab_path,input_texts):
    embedding_dim = 32
    hidden_dim = 128

    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    model = build_model(len(vocab),embedding_dim, hidden_dim, num_classes=7)
    model.load_state_dict(torch.load(model_path))

    x = []
    for input_text in input_texts:
        encoded_text = [vocab.get(char, vocab["unk"]) for char in input_text]
        x.append(encoded_text)

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))
        probabilities = torch.softmax(result, dim=1)
        max_probs, predicted_classes = torch.max(probabilities, 1)

    for i,input_text in enumerate(input_texts):
        class_idx = predicted_classes[i].item()
        prob_value = max_probs[i].item()
        print("输入%s, 预测类别：%d， 概率值%f" %(input_text,class_idx,prob_value))

if __name__ == "__main__":
    main()
    test_strings = ["你nvf我", "w你你df", "rqw你e", "w我你ww", "abcde"]
    predict("model2.pth", "vocab.json", test_strings)
