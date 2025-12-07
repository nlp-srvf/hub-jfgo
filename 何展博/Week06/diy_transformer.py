import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # 线性投影层
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.W_o = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        x: [batch_size, seq_len, embed_dim]
        mask: 可选，[batch_size, seq_len, seq_len] 或 [seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()

        # 1. 线性投影
        Q = self.W_q(x)  # [B, T, E]
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. 分割为多头
        def split_heads(x):
            # [B, T, E] -> [B, H, T, E/H]
            return x.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(Q)  # [B, H, T, E/H]
        K = split_heads(K)
        V = split_heads(V)

        # 3. 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # 应用mask（如果存在）
        if mask is not None:
            # mask: [B, T, T] 或 [T, T] -> [B, 1, T, T]
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [B, 1, T, T]
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # 4. softmax和加权
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, E/H]

        # 5. 合并头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.W_o(attn_output)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.linear1 = nn.Linear(embed_dim, ff_dim)
        self.linear2 = nn.Linear(ff_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        return self.linear2(self.gelu(self.linear1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = FeedForward(embed_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 1. 多头注意力 + 残差连接 + LayerNorm
        attn_output = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 2. 前馈网络 + 残差连接 + LayerNorm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


if __name__ == "__main__":
    # 设置参数
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048

    # ✅ 1. 创建位置编码器（关键添加！）
    pos_enc = PositionalEncoding(embed_dim)

    # 创建Transformer
    transformer = TransformerEncoder(
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim
    )

    # 模拟输入：batch_size=2, seq_len=10
    x = torch.randn(2, 10, embed_dim)

    x = pos_enc(x)

    # 前向传播
    output = transformer(x)

