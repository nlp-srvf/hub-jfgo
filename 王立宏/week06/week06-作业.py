import torch
import torch.nn as nn
import math

'''
使用 PyTorch 实现 transformer 结构
'''


# Embedding层：将输入符号转换为向量，并融合位置和类型信息
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size):
        super(Embeddings, self).__init__()  # 调用父类nn.Module的初始化方法
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)  # 定义词嵌入层，将词汇索引映射到隐藏大小的向量
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)  # 定义token类型嵌入层（如句子A/B）
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)  # 定义位置嵌入层，处理序列位置
        self.LayerNorm = nn.LayerNorm(hidden_size)  # 定义层归一化，稳定训练过程

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)  # 获取输入序列的长度（第1维是序列长度）
        # 以下行有错误：position_ids未定义，应改为torch.arange(seq_length).expand(input_ids.size(0), -1)
        position_ids = self.posistion_ids[:, :seq_length]  # 错误：posistion_ids未初始化，可能是拼写错误，应为预定义的位置id矩阵
        if token_type_ids is not None:  # 如果未提供token_type_ids，则默认为全0
            token_type_ids = torch.zeros_like(input_ids)  # 创建与input_ids相同形状的全0张量
        we = self.word_embeddings(input_ids)  # 计算词嵌入：将输入id转换为向量
        te = self.token_type_embeddings(token_type_ids)  # 计算token类型嵌入
        pe = self.position_embeddings(position_ids)  # 计算位置嵌入
        # 三种embedding相加：融合词、类型和位置信息
        embeddings = we + te + pe
        # 归一化：应用LayerNorm使输出稳定
        embeddings = self.LayerNorm(embeddings)
        return embeddings  # 返回融合后的嵌入向量


# 自注意力机制：计算序列中元素间的关联权重
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()  # 调用父类初始化
        self.num_attention_heads = num_attention_heads  # 设置注意力头数
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 计算每个头的维度
        self.all_head_size = self.num_attention_heads * self.attention_head_size  # 总维度（头数×头维度）
        self.q = nn.Linear(hidden_size, self.all_head_size)  # 定义查询（Query）线性变换层
        self.k = nn.Linear(hidden_size, self.all_head_size)  # 定义键（Key）线性变换层
        self.v = nn.Linear(hidden_size, self.all_head_size)  # 定义值（Value）线性变换层

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # 重塑形状：增加头维度
        x = x.view(*new_x_shape)  # 调整张量形状为(batch_size, seq_len, num_heads, head_size)
        return x.permute(0, 2, 1, 3)  # 转置维度为(batch_size, num_heads, seq_len, head_size)，便于注意力计算

    def forward(self, hidden_states):
        # 线性变换：将输入通过Q、K、V层，并转置为多头格式
        q_layer = self.transpose_for_scores(self.q(hidden_states))  # 变换查询张量
        k_layer = self.transpose_for_scores(self.k(hidden_states))  # 变换键张量
        v_layer = self.transpose_for_scores(self.v(hidden_states))  # 变换值张量
        # 计算注意力分数：Q和K的点积，衡量序列元素间相关性
        attention_scores = torch.matmul(q_layer,
                                        k_layer.transpose(-1, -2))  # 矩阵乘法，得分形状(batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # 缩放得分，防止梯度消失
        # softmax层：将得分转换为概率分布（注意力权重）
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # 沿最后一个维度应用softmax
        # 与v相乘：使用权重加权值张量，得到上下文表示
        context_layer = torch.matmul(attention_probs, v_layer)  # 加权和，形状(batch_size, num_heads, seq_len, head_size)
        # 转置：将多头结果合并回原始格式
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # 转置为(batch_size, seq_len, num_heads, head_size)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # 准备重塑形状：合并头维度
        context_layer = context_layer.view(*new_context_layer_shape)  # 重塑为(batch_size, seq_len, all_head_size)
        return context_layer, attention_probs  # 返回上下文向量和注意力权重（用于可视化）


# 多头注意力机制：包装自注意力，添加残差连接和归一化（但代码不完整）
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()  # 调用父类初始化
        self.output = nn.Linear(hidden_size, hidden_size)  # 定义输出线性层，映射回隐藏大小
        self.LayerNorm = nn.LayerNorm(hidden_size)  # 定义层归一化

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力：但self.self未在__init__中定义，可能应为SelfAttention实例
        self_output, attention_probs = self.self(hidden_states, attention_mask)  # 错误：self.self未初始化，需在__init__中添加
        # 线性层：将自注意力输出通过线性变换
        attention_output = self.output(self_output)  # 线性投影
        # 残差连接和层归一化：添加跳跃连接以提高梯度流动
        attention_output = self.LayerNorm(attention_output + hidden_states)  # 归一化合并结果
        return attention_output, attention_probs  # 返回输出和注意力权重


# 前馈网络（FeedForward）：通过全连接层进行非线性变换
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForward, self).__init__()  # 调用父类初始化
        self.dense = nn.Linear(hidden_size, intermediate_size)  # 定义全连接层：隐藏大小到中间大小
        self.activation = nn.GELU()  # 定义激活函数（GELU，常用于Transformer）
        self.output = nn.Linear(intermediate_size, hidden_size)  # 定义输出全连接层：映射回隐藏大小
        self.LayerNorm = nn.LayerNorm(hidden_size)  # 定义层归一化

    def forward(self, hidden_states):
        # 中间层：通过全连接和激活函数
        intermediate_output = self.dense(hidden_states)  # 线性变换
        intermediate_output = self.activation(intermediate_output)  # 应用GELU激活函数
        # 输出线性层：映射回原始维度
        layer_output = self.output(intermediate_output)  # 线性输出
        # 残差连接和层归一化：合并输入并归一化
        layer_output = self.LayerNorm(layer_output + hidden_states)  # 跳跃连接后归一化
        return layer_output  # 返回前馈网络输出


# Transformer层：组合多头注意力和前馈网络
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(TransformerLayer, self).__init__()  # 调用父类初始化
        self.attention = MultiHeadAttention(hidden_size)  # 定义多头注意力子模块（需完善）
        self.feed_forward = FeedForward(hidden_size, intermediate_size)  # 定义前馈网络子模块

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力层：处理输入并获取注意力输出
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)  # 调用注意力模块
        # 前馈网络：处理注意力输出
        layer_output = self.feed_forward(attention_output)  # 通过前馈网络
        return layer_output, attention_probs  # 返回层输出和注意力权重
