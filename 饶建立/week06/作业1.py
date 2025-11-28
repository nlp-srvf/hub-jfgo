"""
DIY transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """位置编码层"""

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性变换矩阵
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len]
        """
        batch_size, seq_len = query.size(0), query.size(1)

        # 线性变换并分头
        Q = self.W_q(query).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数 QK^T / √d_k
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # 应用mask（如果有）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # softmax得到注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 应用注意力权重到V
        attn_output = torch.matmul(attn_weights, V)

        # 合并多头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )

        # 最终线性变换
        output = self.W_o(attn_output)

        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()  # 或者使用ReLU

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))


class LayerNorm(nn.Module):
    """层归一化"""

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, d_model]
        mask: [batch_size, seq_len, seq_len]
        """
        # 多头自注意力 + 残差连接 + 层归一化
        attn_output, attn_weights = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x, attn_weights


class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, src_mask=None, tgt_mask=None):
        """
        x: 解码器输入 [batch_size, tgt_seq_len, d_model]
        memory: 编码器输出 [batch_size, src_seq_len, d_model]
        """
        # 自注意力（带因果mask）
        self_attn_output, self_attn_weights = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))

        # 交叉注意力
        cross_attn_output, cross_attn_weights = self.cross_attn(
            x, memory, memory, src_mask
        )
        x = self.norm2(x + self.dropout(cross_attn_output))

        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x, self_attn_weights, cross_attn_weights


class TransformerEncoder(nn.Module):
    """Transformer编码器"""

    def __init__(self, vocab_size, d_model, n_heads, d_ff, num_layers,
                 max_seq_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        """
        src: [batch_size, src_seq_len]
        src_mask: [batch_size, src_seq_len, src_seq_len]
        """
        # 词嵌入 + 位置编码
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # 通过所有编码器层
        attention_weights = []
        for layer in self.layers:
            x, attn_weights = layer(x, src_mask)
            attention_weights.append(attn_weights)

        return x, attention_weights


class Transformer(nn.Module):
    """完整的Transformer模型"""

    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, n_heads=8,
                 d_ff=2048, num_layers=6, max_seq_len=5000, dropout=0.1):
        super(Transformer, self).__init__()

        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, n_heads, d_ff, num_layers, max_seq_len, dropout
        )

        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.decoder_pos_encoding = PositionalEncoding(d_model, max_seq_len)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        memory, enc_attention_weights = self.encoder(src, src_mask)

        # 解码器输入处理
        tgt_embedded = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_embedded = self.decoder_pos_encoding(tgt_embedded)
        x = self.dropout(tgt_embedded)

        # 解码器层
        dec_self_attention_weights = []
        dec_cross_attention_weights = []

        for layer in self.decoder_layers:
            x, self_attn, cross_attn = layer(x, memory, src_mask, tgt_mask)
            dec_self_attention_weights.append(self_attn)
            dec_cross_attention_weights.append(cross_attn)

        # 输出层
        output = self.fc_out(x)

        return output, {
            'enc_attention': enc_attention_weights,
            'dec_self_attention': dec_self_attention_weights,
            'dec_cross_attention': dec_cross_attention_weights
        }


# 工具函数
def create_padding_mask(seq, pad_token=0):
    """创建padding mask"""
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask.float()


def create_look_ahead_mask(seq_len):
    """创建因果mask（防止看到未来信息）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    return mask == 0