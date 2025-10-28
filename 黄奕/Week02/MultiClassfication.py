import torch
import torch.nn as nn
import numpy as np
import random
import json

#模型构建
class TorchModel(nn.Module):
    def __init__(self,input_size):
        super(TorchModel,self).__init__()
        self.linear = nn.Linear(input_size,5)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x,y=None):
        y_pred=self.linear(x)
        if y is not None:
            return self.loss(y_pred,y)
        else:
            return y_pred

#随机生成样本
def build_sample():
    x=np.random.random(5)
    y=np.argmax(x)
    return x,y

def build_dataset(num):
    X=[]
    Y=[]
    for i in range(num):
        x,y=build_sample()
        X.append(x)
        Y.append(y)

    # print(X,Y)
    return torch.FloatTensor(X),torch.LongTensor(Y)

#模型评估
def evaluate(model):
    model.eval()
    test_sample_num=100
    x,y=build_dataset(test_sample_num)
    count=[0]*5
    for label in y.tolist():
        count[label]+=1

    print("本次预测集中共有:")
    for i in range(5):
        print("类别%d——%d个\t"%(i,count[i]))

    correct,wrong=0,0
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
    epoch_num=20
    batch_size=20
    train_sample=5000
    input_size=5
    model=TorchModel(input_size)

    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    log=[]

    train_x,train_y=build_dataset(train_sample)
    for epoch in range(epoch_num):
        model.train()
        watch_loss=[]

        for batch_index in range(train_sample//batch_size):
            x=train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y=train_y[batch_index*batch_size:(batch_index+1)*batch_size]

            loss=model.forward(x,y)

            #计算梯度
            loss.backward()
            #更新权重
            optimizer.step()
            #梯度归零
            optimizer.zero_grad()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "MultiClassficationModel.bin")
    print(log)
    return

#预测
def predict(model_path,input_vec):
    input_size=5
    model=TorchModel(input_size)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    # print(model.state_dict())

    model.eval()
    with torch.no_grad():
        logits = model(torch.FloatTensor(input_vec))
        probs = torch.softmax(logits, dim=0)
        pred = torch.argmax(logits).item()

    print("logits :", logits.numpy())
    print("prob   :", probs.numpy())
    print("预测类别 :", pred)



if __name__ == "__main__":
    main()
    input_vec,y=build_sample()
    print()
    print(input_vec)
    predict("MultiClassficationModel.bin",input_vec)
    print("实际类别：",y)