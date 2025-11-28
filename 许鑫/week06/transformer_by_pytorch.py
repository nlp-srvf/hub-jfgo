import torch
import torch.nn as nn
import math


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, mask=None):
        B, S, H = x.shape
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, heads, S, S)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = scores.softmax(dim=-1)
        out = attn @ v  # (B, heads, S, head_dim)
        out = out.transpose(1, 2).reshape(B, S, H)  # (B, S, H)
        return self.out(out)  # 最终线性输出


class FeedForward(nn.Module):
    def __init__(self, hidden_size, ff_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, ff_size)
        self.fc2 = nn.Linear(ff_size, hidden_size)
        self.act = nn.GELU()

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, ff_size, dropout=0.1):
        super().__init__()
        self.attn = MultiHeadSelfAttention(hidden_size, num_heads)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(hidden_size, ff_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attn(x, mask)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))

        return x


if __name__ == '__main__':
    batch = 2
    seq = 5
    hidden = 64
    heads = 8
    ff_hidden = 768
    x = torch.randn(batch, seq, hidden)
    layer = TransformerLayer(hidden, heads, ff_hidden)
    out = layer(x)
    print(out.shape)
