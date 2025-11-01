#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个自行构造的找规律(机器学习)任务
输入一个字符串，根据字符a所在位置进行分类
对比rnn和pooling做法

"""

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        # 各参数解释如下：
        # vector_dim: 每个字符通过embedding层后被转换成的向量维度（embedding_dim），比如5维、8维等。
        # sentence_length: 输入句子的固定长度。如果句子不足则补齐，超长则截断。池化层会用到此长度。
        # vocab: 字符到索引的映射字典，每个字符对应一个唯一编号，用于embedding层的输入。
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  #embedding层
        # RNN 参数解释:
        # nn.RNN(input_size, hidden_size, num_layers=1, nonlinearity='tanh',
        #        bias=True, batch_first=False, dropout=0, bidirectional=False)
        # input_size: 输入x的最后一维大小，也就是每个字符embedding的维数(vector_dim)
        # hidden_size: RNN的隐状态向量维数，本例为vector_dim，表示RNN输出的每一帧的向量长度
        # batch_first: 若为True, 输入和输出的第一个维度是batch_size，否则第一个维度是seq_len
        # 本例 batch_first=True, 输入(x) shape为(batch_size, seq_len, vector_dim)
        # rnn返回两个值: rnn_out (每个时间步的输出, shape=(batch_size, seq_len, hidden_size)) 和 hidden (最后一个时间步的隐状态, shape=(num_layers, batch_size, hidden_size))
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)
        # self.pool = nn.AvgPool1d(sentence_length)   #池化层
        # 将线性层修改为 sentence_length + 1 输出，做分类用（a在各个位置和a不存在）
        self.classify = nn.Linear(vector_dim, sentence_length + 1)     #线性层输出为类别数
        self.loss = nn.functional.cross_entropy  #改为交叉熵损失
        # self.classify = nn.Linear(vector_dim, 1)     #线性层
        # self.activation = torch.sigmoid     #sigmoid归一化函数
        # self.loss = nn.functional.mse_loss  #loss函数采用均方差损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)                      #(batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        #调用rnn
        rnn_out, hidden = self.rnn(x)
        #rnn_out的shape是(batch_size, seq_len, hidden_size)，我们只取最后一个位置的输出
        # 取rnn_out的最后一个时间步的输出作为序列的整体表示。
        # rnn_out的shape是(batch_size, seq_len, hidden_size)
        # rnn_out[:, -1, :] 表示对每个样本，选取序列的最后一个时间步的向量（即最后一列）
        # 这里等价于取每个输入序列处理后的最后一个隐藏状态，用于后续分类
        x = rnn_out[:, -1, :]
        #接线性层做分类
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred, y)   #预测值和真实值计算损失
        else:
            return y_pred                 #输出预测结果

#字符集随便挑了一些字，实际上还可以扩充
#为每个字生成一个标号
#{"a":1, "b":2, "c":3...}
#abc -> [1,2,3]
def build_vocab():
    chars = "abcdefghijklmnopqrstuvwxyz"  #字符集
    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1   #每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab

#随机生成一个样本
#从所有字中选取sentence_length个字
def build_sample(vocab, sentence_length):
    #注意这里用sample，是不放回的采样，每个字母不会重复出现，但是要求字符串长度要小于词表长度
    x = random.sample(list(vocab.keys()), sentence_length)
    #x 为列表，元素为字符串，如['a', 'b', 'c']
    #a在第几位，y的值取vocab的value
    if "a" in x:
        y = x.index("a")   #a在第几位，y的值取vocab的value
    else:
        y = sentence_length
    x = [vocab.get(word, vocab['unk']) for word in x]   #将字转换成序号，为了做embedding        
    return x, y

#建立数据集
#输入需要的样本数量。需要多少生成多少
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)  # 直接添加y，不要用[]包裹，保持1D
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)  # 分类任务使用LongTensor

#建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model

#测试代码
#用来测试每轮模型的准确率
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)   #建立200个用于测试的样本
    print("本次预测集中共有%d个样本"%(len(y)))
    correct, wrong = 0, 0
    #不计算梯度
    with torch.no_grad():
        y_pred = model(x)      #模型预测
        # print(y_pred, "y_pred")
        #zip函数将y_pred和y打包成一个元组，然后遍历每个元组
        for y_p, y_t in zip(y_pred, y):  #与真实标签进行对比
            # torch.argmax(y_p) 的作用是返回 y_p 张量中最大值的索引（即最大概率对应的类别），
            # 这里用来判断模型预测的类别是否等于真实类别
            if int(torch.argmax(y_p)) == int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f"%(correct, correct/(correct+wrong)))  
    return correct/(correct+wrong)


def main():
    #配置参数
    epoch_num = 10        #训练轮数
    batch_size = 20       #每次训练样本个数
    train_sample = 500    #每轮训练总共训练的样本总数
    char_dim = 20         #每个字的维度
    sentence_length = 6   #样本文本长度
    learning_rate = 0.005 #学习率
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length) #构造一组训练样本
            optim.zero_grad()    #梯度归零
            # 解释: loss = model(x, y) 是通过调用模型的 forward 方法来计算损失的。
            # x, y 是通过 build_dataset(batch_size, vocab, sentence_length) 构建得到的小批量输入和标签。
            # 在 TorchModel(nn.Module) 类中，forward 方法定义了模型的前向传播过程：
            # - 当传入 y 时，模型会根据 x 计算预测结果，然后与 y 一起计算 loss（例如均方误差loss）。
            # - 具体实现通常是: 
            #     def forward(self, x, y=None):
            #         y_pred = ... # 基于x得到预测
            #         if y is not None:
            #             return loss_fn(y_pred, y)
            #         else:
            #             return y_pred
            # 因此此处 loss = model(x, y) 就会自动执行 forward(x, y) 并返回损失值。
            loss = model(x, y)   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)   #测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    #保存模型
    torch.save(model.state_dict(), "model_rnn.pth")
    # 保存词表
    writer = open("vocab_rnn.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return

#使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8")) #加载字符表
    model = build_model(vocab, char_dim, sentence_length)     #建立模型
    model.load_state_dict(torch.load(model_path))             #加载训练好的权重
    # print("模型权重：", model.state_dict())
    x = []
    for input_string in input_strings:
        # x.append([vocab[char] for char in input_string])  #将输入序列化
        # 将字符转换为索引，处理不在词表中的字符
        char_indices = [vocab.get(char, vocab.get('unk', 0)) for char in input_string]
        # 截断或填充到固定长度
        if len(char_indices) > sentence_length:
            char_indices = char_indices[:sentence_length]  # 截断
        elif len(char_indices) < sentence_length:
            char_indices = char_indices + [vocab.get('pad', 0)] * (sentence_length - len(char_indices))  # 填充
        x.append(char_indices)
    # print(x, "输入序列")
    model.eval()   #测试模式
    with torch.no_grad():  #不计算梯度
        # torch.LongTensor(x) 的作用是将列表 x（可以是嵌套的整数索引列表）转换为一个 PyTorch 的 LongTensor 类型张量。
        # 在 NLP 场景中，x 通常代表一个或多个样本的字符或单词的索引列表。
        # LongTensor 是 torch 中用于存储 64位整数数据的张量类型，通常 embedding 层的输入都需要是 LongTensor。
        # 以 x = [[1,2,3,4,5,6], [3,4,5,6,7,1]] 为例，torch.LongTensor(x) 会变成 shape=(2,6) 的长整型张量。
        # 这样可以直接作为 embedding 层等神经网络模块的输入。
        result = model.forward(torch.LongTensor(x))  #模型预测，shape为(batch_size, num_classes)
    for i, input_string in enumerate(input_strings):
        # pred_class = torch.argmax(result[i]).item()  # 获取预测类别索引
        # prob = torch.softmax(result[i], dim=0)[pred_class].item()  # 获取该类别的概率
        # print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, pred_class, prob)) #打印结果
        # print("输入：%s, 预测类别：%s, 概率值：%s" % (input_string, torch.argmax(result[i]), result[i])) #打印结果
        pred_class = torch.argmax(result[i]).item()  # 获取预测类别索引
        prob = torch.softmax(result[i], dim=0)[pred_class].item()  # 获取该类别的概率
        # 解释预测结果：如果pred_class < sentence_length，表示'a'在第pred_class位；如果等于sentence_length，表示没有'a'
        if pred_class < sentence_length:
            print("输入：%s, 预测：字符'a'在第%d位, 置信度：%.4f" % (input_string, pred_class, prob))
        else:
            print("输入：%s, 预测：字符串中没有字符'a', 置信度：%.4f" % (input_string, prob))



if __name__ == "__main__":
    # main()
    # test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww"]
    test_strings = ["kijabc", "gijkbc", "gkijad", "kijhd"]
    predict("model_rnn.pth", "vocab_rnn.json", test_strings)
