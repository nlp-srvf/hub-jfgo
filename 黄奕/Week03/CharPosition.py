import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
使用rnn模型训练
判断特定字符在文本中的位置。

"""

class RNNModel(nn.Module):
    def __init__(self,vector_dim,sentence_length,vocab):
        super(RNNModel,self).__init__()
        self.embedding = nn.Embedding(len(vocab),vector_dim,padding_idx=0)
        self.rnn = nn.RNN(vector_dim,vector_dim,bias=False,batch_first=True)
        self.classify = nn.Linear(vector_dim,sentence_length+1)
        self.loss = nn.functional.cross_entropy


    def forward(self,x,y=None):
        x= self.embedding(x)
        # print(x.shape)
        x,h= self.rnn(x)
        # print(x)
        x= x[:,-1,:]
        # print(x)
        y_pred = self.classify(x)
        # print(y_pred.shape)
        y_pred = y_pred.squeeze()
        # print(y_pred.shape)

        if y is not None:
            return self.loss(y_pred,y)
        else:
            return torch.softmax(y_pred,dim=-1)


#构建词表
def build_vocab():
    chars="abcde"

    vocab = {"pad":0}
    for index, char in enumerate(chars):
        vocab[char] = index+1

    vocab["unk"] = len(vocab)
    return vocab

# vocab=build_vocab()
# print(list(vocab))

#随机生成样本
def build_sample(vocab,sentence_length):
    x=[random.choice(list(vocab)) for _ in range(sentence_length)]

    if 'a' in x:
        y=x.index('a')
    else:
        y=sentence_length

    x=[vocab.get(word,vocab['unk']) for word in x]

    return x,y

#分层采样
def build_dataset(sample_length,vocab,sentence_length):
    num_class=sentence_length+1
    per_class=sample_length//num_class
    rem=sample_length%num_class


    dataset_x=[]
    dataset_y=[]

    for cls in range(num_class):
        for i in range(per_class):
            while True:
                x,y=build_sample(vocab,sentence_length)
                if(y==cls):
                    dataset_x.append(x)
                    dataset_y.append(y)
                    break

    for i in range(rem):
        x,y=build_sample(vocab,sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)

    combined = list(zip(dataset_x, dataset_y))
    random.shuffle(combined)
    dataset_x, dataset_y = zip(*combined)
    # print(len(dataset_x))

    return torch.LongTensor(dataset_x),torch.LongTensor(dataset_y)

#建立模型
def build_model(vocab,char_dim,sentence_length):
    model = RNNModel(char_dim,sentence_length,vocab)

    return model


def evaluate(model,vocab,sentence_length):
    model.eval()
    # print("Model_Evaluate")
    x,y = build_dataset(200,vocab,sentence_length)

    # print(x)
    # print(y)
    count=[0]*(sentence_length+1)
    for label in y.tolist():
        count[label]+=1


    print("本次预测集中共有:")
    for i in range(sentence_length+1):
        print("类别%d——%d个\t" % (i, count[i]))

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred=model(x)
        y_pred=np.argmax(y_pred,axis=1)

        for y_p,y_t in zip(y_pred, y):
            if(y_p == y_t):
                correct+=1
            else:
                wrong+=1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    epoch = 10
    batch_size = 20
    train_sample = 500
    char_dim = 20
    sentence_length = 5
    learning_rate = 0.005


    vocab = build_vocab()

    model = build_model(vocab,char_dim,sentence_length)

    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    log=[]
    # print("Model_Train")
    for epoch in range(epoch):
        model.train()
        watch_loss=[]

        for batch in range(int(train_sample/batch_size)):
            x,y = build_dataset(train_sample,vocab,sentence_length)


            loss=model(x,y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model,vocab,sentence_length)
        log.append({"epoch":epoch+1, "loss":np.mean(watch_loss), "acc":acc})

    torch.save(model.state_dict(),"CharPositionModel.pth")

    #保存词表
    writer = open("vocab.json","w",encoding="utf-8")
    writer.write(json.dumps(vocab,ensure_ascii=False,indent=2))
    writer.close()
    return

def predict(model,vocab,input_strings):
    char_dim=20
    sentence_length=5
    vocab = json.load(open("vocab.json","r",encoding="utf-8"))
    model = build_model(vocab,char_dim,sentence_length)
    model.load_state_dict(torch.load("CharPositionModel.pth", map_location="cpu", weights_only=True))
    x=[]
    for input_string in input_strings:
        x.append([vocab.get(word, vocab['unk']) for word in input_string])
    print(x)

    model.eval()
    with torch.no_grad():
        logits = model(torch.LongTensor(x))
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(logits,dim=-1)

        # print("logits :", logits.numpy())
        # print("prob   :", probs.numpy())
        print("预测类别 :", pred)


if __name__ == "__main__":
    main()
    test_strings = ["fnaje", "wbsfa", "rwdeg", "nkkww"]
    predict("model.pth", "vocab.json", test_strings)