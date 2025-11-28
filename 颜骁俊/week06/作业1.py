import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        参数说明:
        d_model: 词嵌入的维度 (例如 512)
        num_heads: 多头注意力的头数 (例如 8)
        d_ff: 前馈神经网络的隐藏层维度 (通常是 d_model 的 4 倍，例如 2048)
        dropout:以此概率丢弃神经元
        """
        super().__init__()
        
        # --- 子层 1: 多头自注意力机制 ---
        # batch_first=True 让输入维度变为 [Batch, Seq_Len, Feature]
        self.self_attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, 
                                               dropout=dropout, batch_first=True)
        
        # --- 子层 2: 前馈神经网络 (FFN) ---
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        
        # --- 归一化层 (LayerNorm) ---
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # --- Dropout 层 ---
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        """
        src: 输入张量，形状为 [Batch, Seq_Len, d_model]
        src_mask: (可选) 掩码，用于忽略 padding 或未来的 token
        """
        
        # --- 步骤 1: 自注意力计算 (Self-Attention) ---
        # MultiheadAttention 的输入需要 (query, key, value)
        # 在自注意力中，Q, K, V 都是 src
        attn_output, _ = self.self_attn(src, src, src, attn_mask=src_mask)
        
        # --- 步骤 2: 残差连接 + 归一化 (Add & Norm) ---
        # 公式: LayerNorm(x + Dropout(Sublayer(x)))
        src = src + self.dropout(attn_output) # 残差连接 (Residual Connection)
        src = self.norm1(src)                 # 层归一化
        
        # --- 步骤 3: 前馈网络 (Feed Forward) ---
        # 结构: Linear -> ReLU/GELU -> Dropout -> Linear
        ff_output = self.linear2(self.dropout(F.relu(self.linear1(src))))
        
        # --- 步骤 4: 残差连接 + 归一化 (Add & Norm) ---
        src = src + self.dropout(ff_output)   # 残差连接
        src = self.norm2(src)                 # 层归一化
        
        return src

# --- 测试代码 ---
if __name__ == "__main__":
    # 定义超参数
    BATCH_SIZE = 2
    SEQ_LEN = 10
    D_MODEL = 512
    NUM_HEADS = 8
    D_FF = 2048

    # 1. 实例化我们的 Transformer 层
    layer = TransformerBlock(d_model=D_MODEL, num_heads=NUM_HEADS, d_ff=D_FF)

    # 2. 创建一个随机输入 (Batch, Seq_Len, d_model)
    x = torch.randn(BATCH_SIZE, SEQ_LEN, D_MODEL)

    # 3. 前向传播
    output = layer(x)

    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}") # 应该保持不变 [2, 10, 512]
