import torch
import torch.nn as nn
import math
import numpy as np
from transformers import BertModel

'''
通过PyTorch实现Bert结构
模型文件下载 https://huggingface.co/models
'''

# 加载预训练模型和权重
bert = BertModel.from_pretrained(r"D:\nlp\bert-base-chinese", return_dict=False)
state_dict = bert.state_dict()
bert.eval()

# 示例输入
x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
torch_x = torch.LongTensor([x])   # PyTorch形式输入
seqence_output, pooler_output = bert(torch_x)
print("PyTorch原生Bert输出形状:", seqence_output.shape, pooler_output.shape)


# 查看所有的权值矩阵名称
# print(bert.state_dict().keys())

class bertModel(nn.Module):
    def __init__(self, state_dict):
        super(bertModel, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12  # 注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # Embedding部分 将与训练模型的初始权重赋值
        # word_embeddings
        self.word_embeddings=nn.Embedding.from_pretrained(state_dict["embeddings.word_embeddings.weight"],freeze=False)
        # token_embeddings 将bert与训练模型的初始权重赋值
        self.token_embedding =nn.Embedding.from_pretrained(state_dict["embeddings.token_type_embeddings.weight"],freeze=False)
        # position_embeddings
        self.position_embedding = nn.Embedding.from_pretrained(state_dict["embeddings.position_embeddings.weight"],freeze=False)
        # 过一个归一化层
        self.embeddings_layer_norm = torch.nn.LayerNorm(self.hidden_size)
        # 加载LayerNorm的权重和偏置
        self.embeddings_layer_norm.weight.data = state_dict["embeddings.LayerNorm.weight"]
        self.embeddings_layer_norm.bias.data = state_dict["embeddings.LayerNorm.bias"]

        # Transformer部分，有多层
        self.transformer_layers = nn.ModuleList()
        for i in range(self.num_layers):
            # 多层单独使用一个方法
            layer=TransformerLayer(
                hidden_size=self.hidden_size,
                num_attention_heads=self.num_attention_heads,
                state_dict=state_dict,
                layer_idx=i
            )
            self.transformer_layers.append(layer)

        # Pooler层
        self.pooler_dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.pooler_dense.weight.data = state_dict["pooler.dense.weight"]
        self.pooler_dense.bias.data = state_dict["pooler.dense.bias"]

    def embedding_forward(self, x):
        # x.shape = [batch_size, max_len]
        # 因为输入是batch_size 为1
        max_len = x.shape[1]

        # 过embedding层
        we = self.word_embeddings(x)  # [max_len, hidden_size]:[4,768]

        # 位置embedding 输入是下标位置
        # position_ids也可以直接定义一个512长度的
        position_ids = torch.arange(max_len, dtype=torch.long) #[0,1,2,3..maxlen-1]
        pe = self.position_embedding(position_ids)  # shape: [max_len, hidden_size]:[4,768]

        # 句子embedding 区分句子 单输入的情况下为[0, 0, 0, 0]
        token_type_ids = torch.zeros_like(x)
        te = self.token_embedding(token_type_ids)  # shape: [max_len, hidden_size]:[4,768]

        # 三者相加并进行Layer Normalization
        embedding = we + pe + te
        # 过一个归一化层
        embedding = self.embeddings_layer_norm(embedding) # shape: [max_len, hidden_size]:[4,768]
        return embedding

    def forward(self, x):
        # Embedding层
        x = self.embedding_forward(x)  # shape: [max_len, hidden_size]:[4,768]

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x)  # shape: [batch_size, max_len, hidden_size]

        # 提取[CLS] token的输出作为pooler output
        pooler_output = self.pooler_dense(x[:, 0, :])  # shape: [batch_size, hidden_size]
        pooler_output = torch.tanh(pooler_output)

        return x, pooler_output  # sequence_output: [batch_size, max_len, hidden_size], pooler_output: [batch_size, hidden_size]


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, state_dict, layer_idx):
        super(TransformerLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        # dk 多投机制中的投数
        self.attention_head_size = hidden_size // num_attention_heads  # 每个头的维度（768//12=64）

        # Self-Attention层
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        # 加载权重
        self.query.weight.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.query.weight"]
        self.query.bias.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.query.bias"]
        self.key.weight.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.key.weight"]
        self.key.bias.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.key.bias"]
        self.value.weight.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.value.weight"]
        self.value.bias.data = state_dict[f"encoder.layer.{layer_idx}.attention.self.value.bias"]

        # Attention输出投影层
        self.attention_output = nn.Linear(hidden_size, hidden_size)
        self.attention_output.weight.data = state_dict[f"encoder.layer.{layer_idx}.attention.output.dense.weight"]
        self.attention_output.bias.data = state_dict[f"encoder.layer.{layer_idx}.attention.output.dense.bias"]
        # 归一化层
        self.attention_layer_norm = nn.LayerNorm(hidden_size)
        self.attention_layer_norm.weight.data = state_dict[
            f"encoder.layer.{layer_idx}.attention.output.LayerNorm.weight"]
        self.attention_layer_norm.bias.data = state_dict[f"encoder.layer.{layer_idx}.attention.output.LayerNorm.bias"]

        # Feed Forward层
        # 过一个线性层 lxh -> hx4h -> lx4h
        self.intermediate = nn.Linear(hidden_size, 4 * hidden_size)
        # 再将lx4h -> 4hxh -> lxh
        self.output = nn.Linear(4 * hidden_size, hidden_size)
        # 加载Feed Forward层权重
        self.intermediate.weight.data = state_dict[f"encoder.layer.{layer_idx}.intermediate.dense.weight"]
        self.intermediate.bias.data = state_dict[f"encoder.layer.{layer_idx}.intermediate.dense.bias"]
        self.output.weight.data = state_dict[f"encoder.layer.{layer_idx}.output.dense.weight"]
        self.output.bias.data = state_dict[f"encoder.layer.{layer_idx}.output.dense.bias"]

        # Feed Forward层的归一化层
        self.ff_layer_norm = torch.nn.LayerNorm(hidden_size)
        self.ff_layer_norm.weight.data = state_dict[f"encoder.layer.{layer_idx}.output.LayerNorm.weight"]
        self.ff_layer_norm.bias.data = state_dict[f"encoder.layer.{layer_idx}.output.LayerNorm.bias"]

    def self_attention(self, x):
        # x.shape = [batch_size, max_len, hidden_size]
        # 输入是embedding层的输出
        batch_size, max_len, hidden_size = x.shape
        # 获取q,k,v的值 shape:[max_len,hidden_size]
        q=self.query(x)
        k=self.key(x)
        v=self.value(x)

        # 多投机制
        # 将q, k, v拆分为多个头后再拼接
        q = q.view(batch_size, max_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        k = k.view(batch_size, max_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)
        v = v.view(batch_size, max_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2)

        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # shape: [batch_size, num_heads, max_len, max_len]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # 计算注意力权重
        attention_weights = torch.nn.functional.softmax(attention_scores,dim=-1)  # shape: [batch_size, num_heads, max_len, max_len]
        # 应用注意力权重到value上
        attention_output = torch.matmul(attention_weights,v)  # shape: [batch_size, num_heads, max_len, attention_head_size]
        # 将多个头的输出拼接起来
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, max_len, hidden_size)
        # 最终的线性投影
        attention_output = self.attention_output(attention_output)  # shape: [batch_size, max_len, hidden_size]

        return attention_output

    def feed_forward(self,x):
        # 过一层线性层
        x=self.intermediate(x) #shape:[max_len,4*hidden_size]
        # 过一个激活层
        x=torch.nn.functional.gelu(x)
        # 第二个线性层
        x=self.output(x) #shape:[max_len,hidden_size]
        return x

    def forward(self,x):
        # embedding层的输出是x 作为self_attention的输入
        self_attention_output = self.self_attention(x)
        # 残差机制之后过一层归一化
        x = self.attention_layer_norm(x + self_attention_output)
        # feed forward
        ff_output = self.feed_forward(x)
        # 残差
        x = self.ff_layer_norm(x+ff_output)
        return x



# 测试自制的PyTorch Bert模型
db = bertModel(state_dict)
diy_sequence_output, diy_pooler_output = db(torch_x)
#torch
torch_sequence_output, torch_pooler_output = bert(torch_x)

print(diy_sequence_output)
print(torch_sequence_output)
