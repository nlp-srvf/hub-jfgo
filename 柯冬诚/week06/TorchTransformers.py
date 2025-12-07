import torch
import torch.nn as nn
import math

'''
基于PyTorch实现transformer
'''

# Embedding层
class Embeddings(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_position_embeddings, type_vocab_size):
        super(Embeddings, self).__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = self.posistion_ids[:, :seq_length]
        if token_type_ids is not None:
            token_type_ids = torch.zeros_like(input_ids)

        we = self.word_embeddings(input_ids)
        te = self.token_type_embeddings(token_type_ids)
        pe = self.position_embeddings(position_ids)
        # 三种embedding相加
        embeddings = we + te + pe
        # 归一化
        embeddings = self.LayerNorm(embeddings)

        return embeddings


# 自注意力机制
class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads):
        super(SelfAttention, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.q = nn.Linear(hidden_size, self.all_head_size)
        self.k = nn.Linear(hidden_size, self.all_head_size)
        self.v = nn.Linear(hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        # 线性变换
        q_layer = self.transpose_for_scores(self.q(hidden_states))
        k_layer = self.transpose_for_scores(self.k(hidden_states))
        v_layer = self.transpose_for_scores(self.v(hidden_states))

        # 计算注意力分数
        attention_scores = torch.matmul(q_layer, k_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # softmax层
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 与 v 相乘
        context_layer = torch.matmul(attention_probs, v_layer)

        # 转置
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, attention_probs


# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.output = nn.Linear(hidden_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力
        self_output, attention_probs = self.self(hidden_states, attention_mask)

        # 线性层 Liner(Attention(Q,K,V))
        attention_output = self.output(self_output)

        # 残差连接和层归一化 LayerNorm(Xembedding+ Xattention)
        attention_output = self.LayerNorm(attention_output + hidden_states)

        return attention_output, attention_probs


# feed-forward
class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(FeedForward, self).__init__()
        self.linearLayer = nn.Linear(hidden_size, intermediate_size)
        self.activation = nn.GELU()
        self.output = nn.Linear(intermediate_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        # 第一个线性层 Liner(x)
        intermediate_output = self.linearLayer(hidden_states)

        # gelu激活函数 gelu(Liner(x))
        intermediate_output = self.activation(intermediate_output)

        # 外层线性层 Liner(gelu(Liner(x)))
        layer_output = self.output(intermediate_output)

        # 残差连接和层归一化 LayerNorm(X forward+ Xattention)
        layer_output = self.LayerNorm(layer_output + hidden_states)

        return layer_output


# transformer层
class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

    def forward(self, hidden_states, attention_mask=None):
        # 自注意力层
        attention_output, attention_probs = self.attention(hidden_states, attention_mask)

        # 前馈网络feed-forward
        layer_output = self.feed_forward(attention_output)

        return layer_output, attention_probs
