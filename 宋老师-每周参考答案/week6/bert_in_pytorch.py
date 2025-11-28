import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SimpleBERT(nn.Module):
    def __init__(self, vocab_size=30522, hidden_size=768, num_hidden_layers=12, 
                 num_attention_heads=12, intermediate_size=3072, 
                 max_position_embeddings=512, hidden_dropout_prob=0.1, 
                 attention_probs_dropout_prob=0.1, type_vocab_size=2):
        super(SimpleBERT, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # 嵌入层
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, hidden_size)
        self.LayerNorm = nn.LayerNorm(hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList([
            SimpleBERTEncoderLayer(hidden_size, num_attention_heads, intermediate_size,
                           hidden_dropout_prob, attention_probs_dropout_prob)
            for _ in range(num_hidden_layers)
        ])
        
    
    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            token_type_ids: [batch_size, seq_len]
            position_ids: [batch_size, seq_len]
        """
        batch_size, seq_len = input_ids.shape
        
        # 创建位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, seq_len)
        
        # 创建token类型ID
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # 嵌入层
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # 组合嵌入
        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        
        # 通过编码器层
        hidden_states = embeddings
        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states)
        
        return hidden_states


class SimpleBERTEncoderLayer(nn.Module):
    """简化的BERT编码器层"""
    
    def __init__(self, hidden_size, num_attention_heads, intermediate_size,
                 hidden_dropout_prob, attention_probs_dropout_prob):
        super(SimpleBERTEncoderLayer, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # 自注意力机制
        self.self_attention = SimpleMultiHeadAttention(
            hidden_size, num_attention_heads, attention_probs_dropout_prob
        )
        self.attention_output_dropout = nn.Dropout(hidden_dropout_prob)
        self.attention_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
        
        # 前馈网络
        self.intermediate = SimpleFeedForward(hidden_size, intermediate_size)
        self.output_dropout = nn.Dropout(hidden_dropout_prob)
        self.output_layer_norm = nn.LayerNorm(hidden_size, eps=1e-12)
    
    def forward(self, hidden_states):
        # 自注意力子层
        attention_output = self.self_attention(hidden_states)
        attention_output = self.attention_output_dropout(attention_output)
        attention_output = self.attention_layer_norm(attention_output + hidden_states)
        
        # 前馈网络子层
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.output_dropout(intermediate_output)
        layer_output = self.output_layer_norm(intermediate_output + attention_output)
        
        return layer_output


class SimpleMultiHeadAttention(nn.Module):
    """简化的多头注意力机制"""
    
    def __init__(self, hidden_size, num_attention_heads, dropout_prob):
        super(SimpleMultiHeadAttention, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads
        
        # 查询、键、值投影
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_prob)
        self.dense = nn.Linear(hidden_size, hidden_size)
    
    def transpose_for_scores(self, x):
        """转置张量以计算注意力分数"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape
        
        # 线性投影
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 注意力概率
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 上下文层
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        # 输出投影
        attention_output = self.dense(context_layer)
        
        return attention_output


class SimpleFeedForward(nn.Module):
    """简化的前馈网络"""
    
    def __init__(self, hidden_size, intermediate_size):
        super(SimpleFeedForward, self).__init__()
        
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.intermediate_act_fn = nn.GELU()
    
    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.dense2(hidden_states)
        return hidden_states


# 使用示例
if __name__ == "__main__":
    # 创建简化BERT模型
    bert = SimpleBERT(
        vocab_size=10000,  # 使用较小的词汇表
        hidden_size=512,   # 使用较小的隐藏层
        num_hidden_layers=6,  # 使用较少的层数
        num_attention_heads=8,
        intermediate_size=1024,
        max_position_embeddings=512
    )
    
    # 创建示例输入
    batch_size = 2
    seq_len = 128
    
    input_ids = torch.randint(0, 10000, (batch_size, seq_len))
    
    # 前向传播
    outputs = bert(input_ids)
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输出形状: {outputs.shape}")
    
    # 测试单个注意力头
    print("\n测试简化多头注意力:")
    multi_head_attn = SimpleMultiHeadAttention(hidden_size=512, num_attention_heads=8, dropout_prob=0.1)
    test_input = torch.randn(batch_size, seq_len, 512)
    attn_output = multi_head_attn(test_input)
    print(f"注意力输入形状: {test_input.shape}")
    print(f"注意力输出形状: {attn_output.shape}")
    
    # 测试单个编码器层
    print("\n测试简化编码器层:")
    encoder_layer = SimpleBERTEncoderLayer(
        hidden_size=512, 
        num_attention_heads=8, 
        intermediate_size=1024,
        hidden_dropout_prob=0.1, 
        attention_probs_dropout_prob=0.1
    )
    layer_output = encoder_layer(test_input)
    print(f"编码器层输入形状: {test_input.shape}")
    print(f"编码器层输出形状: {layer_output.shape}")