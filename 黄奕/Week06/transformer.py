import torch
import torch.nn as nn
import math

class Embedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, max_len=512):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.seg = nn.Embedding(2, hidden_size, padding_idx=0)
        self.pos = nn.Embedding(max_len, hidden_size)

    def forward(self, input_ids, seg_ids=None):
        b, s = input_ids.size()
        pos = torch.arange(s, dtype=torch.long, device=input_ids.device).unsqueeze(0).expand(b, -1)
        if seg_ids is None:
            seg_ids = torch.zeros_like(input_ids)
        return self.tok(input_ids) + self.seg(seg_ids) + self.pos(pos)


class SelfAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        assert hidden_size % num_heads == 0
        self.h = num_heads
        self.d_k = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        b, s, d = x.size()
        qkv = self.qkv(x).view(b, s, 3, self.h, self.d_k).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_k)

        attn = torch.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).contiguous().view(b, s, d)
        return self.out(out)


class FeedForward(nn.Module):
    def __init__(self, hidden_size, intermediate_size, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, intermediate_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_size, hidden_size)
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, hidden_size=768, num_heads=12, intermediate_size=3072):
        super().__init__()
        self.emb = Embedding(vocab_size, hidden_size)
        self.attn = SelfAttention(hidden_size, num_heads)
        self.ffn = FeedForward(hidden_size, intermediate_size)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, input_ids):
        mask = (input_ids != 0).unsqueeze(1).unsqueeze(2)
        x = self.emb(input_ids)
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


if __name__ == "__main__":
    model = TransformerModel(1000, 128, 8, 512)
    tok = torch.randint(0, 1000, (4, 20))
    out = model(tok)
    print(out.shape)