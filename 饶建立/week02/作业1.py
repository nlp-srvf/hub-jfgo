"""
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，哪个维度的值最大，就属于哪个类别
"""
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

# 创建模型
class Torchmodel(nn.Module):
    def __init__(self,input_size,output_size):
        super(Torchmodel, self).__init__()
        self.linear=nn.Linear(input_size,output_size)
        self.loss=nn.CrossEntropyLoss()     # 交叉熵自带softmax

    def forward(self,x,y=None):
        y_pre=self.linear(x)
        if y is not None:
            loss=self.loss(y_pre,y)
            return y_pre,loss
        else:
            return y_pre


# 随机生成训练样本，构造数据
class SampleDataset(Dataset):
    def __init__(self,num_samples,nun_vector):
        self.nun_vector = nun_vector
        self.num_samples=num_samples


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        x=np.random.randn(self.nun_vector).astype('float32')
        y=np.argmax(x)
        return torch.from_numpy(x),y


class MyDataset(Dataset):
    def __init__(self,num_samples,input_dim,num_classes):
        """
            num_samples: 样本数量
            input_dim: 输入维度
            num_classes: 类别数量
        """
        self.num_samples=num_samples
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.data=np.random.randn(num_samples,input_dim).astype('float32')
        self.labels=np.random.randint(0,num_classes,size=num_samples).astype('int32')

        # 转换张量
        self.data=torch.tensor(self.data,dtype=torch.float32)
        self.labels=torch.tensor(self.labels,dtype=torch.long)

        # self.data=torch.from_numpy(self.data)
        # self.labels=torch.from_numpy(self.labels)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx],self.labels[idx]






def train_model_process(model,train_dataloader,num_epochs,model_path):
    # 设定训练设备
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # 指定优化器,损失函数
    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)

    # 将模型放入设备中
    model=model.to(device)

    for epoch in range(num_epochs):
        train_num=0
        train_corrects = 0
        watch_loss = []
        for batch_idx, (batch_x, batch_y) in enumerate(train_dataloader):
            # 将训练数据放入设备中
            b_x=batch_x.to(device)
            b_y=batch_y.to(device)

            model.train()
            # 前向传播,计算损失
            y_pre,loss=model(b_x,b_y)
            pre_label=torch.argmax(y_pre,dim=1)

            # 反向传播计算梯度，参数更新，清空梯度（不清空会累加）
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            # 记录损失
            watch_loss.append(loss.item())
            train_corrects+=torch.sum(pre_label==b_y).item()
            train_num += b_x.size(0)
        print(f"epoch:{epoch},loss:{np.mean(watch_loss)},acc:{train_corrects/train_num}")
        torch.save(model.state_dict(),model_path)




def predict(model_path,test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model.load_state_dict(torch.load(model_path))
    criterion = nn.CrossEntropyLoss()
    model.eval()
    test_loss=[]
    test_corrects=0
    test_num=0
    with torch.no_grad():
        for batch_idx, (batch_x, batch_y) in enumerate(test_dataloader):
            b_x=batch_x.to(device)
            b_y=batch_y.to(device)
            y_pre=model(b_x)
            pre_label=torch.argmax(y_pre,dim=1)
            # 记录损失
            loss=criterion(y_pre,b_y)
            test_loss.append(loss.item())
            test_corrects+=torch.sum(pre_label==b_y).item()
            test_num+=b_x.size(0)
        print(f"loss:{np.mean(test_loss)},acc:{test_corrects/test_num}")








if __name__=="__main__":
    # windows加载多进程必须的
    torch.multiprocessing.freeze_support()

    model=Torchmodel(5,5)

    train_data=SampleDataset(num_samples=10000,nun_vector=5)

    train_dataloader = DataLoader(
        train_data,
        batch_size=100,
        shuffle=True,
        num_workers=2,  # 多进程加载数据
        pin_memory=True  # 加速GPU传输数据
    )
    model_path="./model.pth"
    train_model_process(model,train_dataloader,20,model_path)

    print("="*10+"predict"+"="*10)

    test_data = SampleDataset(num_samples=10000, nun_vector=5)

    test_dataloader = DataLoader(
        test_data,
        batch_size=100,
        shuffle=True,
        num_workers=2,  # 多进程加载数据
        pin_memory=True  # 加速GPU传输数据
    )

    predict(model_path, test_dataloader)








