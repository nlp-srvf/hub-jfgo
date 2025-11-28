# 作业1：计算Bert参数大小
# 层数 (L)：12
# 隐藏层维度 (H)：768
# 注意力头数 (A)：12
# 词表大小 (V)：~30,522


# 1.嵌入层
# （30522词表+2段落+512位置）*768=23.8M
# 2.编码器层12个
# ①自注意力机制层（QKV-scores-softmax-dropout-加权求和-拼接多头-line投影输出）
# 3 * (768 * (768/12)) * 12【QKV】+768 * 768【line】=2.36M
# ②前向反馈层（2个全连接层，X-line1-action-dropout-line2）
# 768 * (4 * 768)+(4 * 768) * 768=4.72M
# ③归一化层（2个归一化层做残差，每个2个参数向量）
# 2*2*768=3000
# （2.36M+4.72M+3000）*12=84.9M
# 3.输出层
# ①池化层（line+激活）
# 768 * 768=0.59M
# ②语言模型头（line+归一化）
# 768 * 768+2*768=0.59M
# ③分类器头
# 768*num_labels（忽略）

# 总计：23.8+84.9+0.59*2=110M


# 作业2：pytorch实现BERT结构
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
"""
实现BERT整体架构
1.嵌入层ok：embedding（词+位置+段落）+layerNorm+dropout
2.编码层
    2-1多头自注意力机制（重点）：QKV(三个linear)-拆头-Q*KT/dk-softmax-dropout-*V-拼接多头-line
    2-2前向反馈ok：line1-relu-dropout-line2
    2-3归一化层两个:Norm【2-1+dropout（2-1）】=y  → Norm 【y+dropout（2-2（y））】
3.Pooler层ok：linear+tanu
总：嵌入层+12个编码层+Pooler层
"""


class BertEmbedding(nn.module):
    # 1.嵌入层
    def __init__(self,vocab):
        super(BertEmbedding, self).__init__()
        self.vocab_embedding=nn.Embedding(vocab,786)
        self.position_embedding = nn.Embedding(512, 786)
        self.para_embedding = nn.Embedding(2, 786)
        self.norm_layer=nn.LayerNorm(786)
        self.dropout=nn.Dropout(0.1)
    def forward(self,input):
        vocab_embedding= self.vocab_embedding(input)
        position_embedding=self.position_embedding(input)
        para_embedding=self.para_embedding(input)
        embeddings=vocab_embedding+position_embedding+para_embedding

        embeddings = self.norm_layer(embeddings)
        embeddings = self.dropout(embeddings)

        return  embeddings
class BearMutitSelfAttention(nn.Module):
    # 2-1多头注意力机制
    def __init__(self):
        super(BearMutitSelfAttention, self).__init__()
        self.lineQ=nn.Linear(512,512)
        self.lineK = nn.Linear(512, 512)
        self.lineV = nn.Linear(512, 512)

        self.head=12
        self.hiddent=786
        self.dk=self.hiddent/self.head

        self.dropout=nn.Dropout(0.1)
        self.line=nn.Linear(512,512)

    def forward(self,x):
        # x.size() :bitchsize,length,786
        batch_size, seq_len, d_model = x.size()

        Q=self.lineQ(x)
        K = self.lineK(x)
        V = self.lineV(x)

        QMutit=Q.view(batch_size, seq_len,self.head,self.dk).transpose(1,2)
        KMutit=K.view(batch_size, seq_len,self.head,self.dk).transpose(1,2)
        VMutit=V.view(batch_size, seq_len,self.head,self.dk).transpose(1,2)

        socres=torch.matmul(QMutit, K.transpose(-1, -2))/math.sqrt(self.dk)
        attention=F.softmax(socres,dim=-1)
        attention=self.dropout(attention)

        attention=torch.matmul(attention,VMutit)
        # attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        attention = attention.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        output=self.line(attention)

        return output
class BearFeedForward(nn.module):
    # 2-2前向反馈
    def __init__(self):
        super(BearFeedForward, self).__init__()
        self.line1=nn.Linear(786,786*4)
        self.action=nn.GELU()
        self.dropout=nn.Dropout(0.1)
        self.line2=nn.Linear(786*4,786)
    def forward(self,input):
        line1=self.line1(input)
        line1=self.action(line1)
        line1 = self.dropout(line1)
        line1 = self.line2(line1)

        return  line1

class BearEncoder(nn.module):
    # 2-3编码层实现，加两个归一化
    def __init__(self):
        super(BearEncoder, self).__init__()
        self.mutui_self_attention=BearMutitSelfAttention()
        self.feed_forward=BearFeedForward()

        self.Norm1=nn.LayerNorm(512)
        self.Norm2=nn.LayerNorm(512)
        self.dropout=nn.Dropout(0.1)

    def forward(self,input):
        res=input
        x1=self.Norm1(res+self.dropout(self.mutui_self_attention(input)))

        res=x1
        x2 = self.Norm1(res + self.dropout(self.feed_forward(input)))

        return  x2

class BearEncoders(nn.Module):
    # 2. 12层编码器实现
    def __init__(self):
        super(BearEncoders, self).__init__()
        self.layers=nn.modulelist([
            BearEncoder(
            )  for _ in range(12)

        ])
    def forward(self,input):
        for  layer in enumerate(self.layers):
            input=layer(input)

        return  input

class BearPooler(nn.Module):
    # 3.Pooler层
    def __init__(self):
        super(BearPooler, self).__init__()
        self.line1=nn.Linear(512,512)
        self.action=nn.Tanh()
    def forward(self,input):
        line1=self.action(self.line1(input))
        return line1

class Bear(nn.module):
    def __init__(self, vocab_size):
        super(Bear, self).__init__()
        self.BertEmbedding = BertEmbedding(vocab_size)
        self.BearEncoders = BearEncoders()
        self.BearPooler = BearPooler()
    def forward(self,x):
        x=self.BertEmbedding(x)
        x = self.BearEncoders(x)
        x = self.BearPooler(x)
        return  x

if __name__ == '__main__':
    vocab_size = 30000
    model = Bear(vocab_size=vocab_size)


