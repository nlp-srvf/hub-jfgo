import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from transformers import BertModel


class DiyBertTorch:
    def __init__(self, state_dict):
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1  # 单层Transformer
        self.load_weights(state_dict)

    def load_weights(self, state_dict):
        # embedding部分
        self.word_embeddings = nn.Parameter(state_dict["embeddings.word_embeddings.weight"])
        self.position_embeddings = nn.Parameter(state_dict["embeddings.position_embeddings.weight"])
        self.token_type_embeddings = nn.Parameter(state_dict["embeddings.token_type_embeddings.weight"])
        self.embeddings_layer_norm_weight = nn.Parameter(state_dict["embeddings.LayerNorm.weight"])
        self.embeddings_layer_norm_bias = nn.Parameter(state_dict["embeddings.LayerNorm.bias"])

        # 单层transformer参数
        i = 0
        self.q_w = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.query.weight"])
        self.q_b = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.query.bias"])
        self.k_w = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.key.weight"])
        self.k_b = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.key.bias"])
        self.v_w = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.value.weight"])
        self.v_b = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.self.value.bias"])
        self.attention_output_weight = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.output.dense.weight"])
        self.attention_output_bias = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.output.dense.bias"])
        self.attention_layer_norm_w = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"])
        self.attention_layer_norm_b = nn.Parameter(state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"])
        self.intermediate_weight = nn.Parameter(state_dict[f"encoder.layer.{i}.intermediate.dense.weight"])
        self.intermediate_bias = nn.Parameter(state_dict[f"encoder.layer.{i}.intermediate.dense.bias"])
        self.output_weight = nn.Parameter(state_dict[f"encoder.layer.{i}.output.dense.weight"])
        self.output_bias = nn.Parameter(state_dict[f"encoder.layer.{i}.output.dense.bias"])
        self.ff_layer_norm_w = nn.Parameter(state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"])
        self.ff_layer_norm_b = nn.Parameter(state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"])

        # pooler层
        self.pooler_dense_weight = nn.Parameter(state_dict["pooler.dense.weight"])
        self.pooler_dense_bias = nn.Parameter(state_dict["pooler.dense.bias"])

    def get_embedding(self, embedding_matrix, x):
        return embedding_matrix[x]

    def embedding_forward(self, x):
        # word embedding
        we = self.get_embedding(self.word_embeddings, x)

        # position embedding
        position_ids = torch.arange(x.size(0), dtype=torch.long)
        pe = self.get_embedding(self.position_embeddings, position_ids)

        # token type embedding (全0)
        token_type_ids = torch.zeros_like(x)
        te = self.get_embedding(self.token_type_embeddings, token_type_ids)

        embedding = we + pe + te
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)
        return embedding

    def layer_norm(self, x, weight, bias, eps=1e-12):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x = (x - mean) / torch.sqrt(var + eps)
        return weight * x + bias

    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        max_len, hidden_size = x.shape
        x = x.view(max_len, num_attention_heads, attention_head_size)
        x = x.permute(1, 0, 2)  # [num_heads, max_len, head_size]
        return x

    def self_attention(self, x, q_w, q_b, k_w, k_b, v_w, v_b,
                       attention_output_weight, attention_output_bias,
                       num_attention_heads, hidden_size):
        attention_head_size = hidden_size // num_attention_heads

        # Q, K, V 投影
        q = F.linear(x, q_w.T, q_b)  # [max_len, hidden_size]
        k = F.linear(x, k_w.T, k_b)  # [max_len, hidden_size]
        v = F.linear(x, v_w.T, v_b)  # [max_len, hidden_size]

        # 多头分割
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)  # [num_heads, max_len, head_size]
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)

        # 注意力计算
        attention_scores = torch.matmul(q, k.transpose(-1, -2))  # [num_heads, max_len, max_len]
        attention_scores = attention_scores / math.sqrt(attention_head_size)

        attention_probs = F.softmax(attention_scores, dim=-1)

        # 注意力加权和
        context_layer = torch.matmul(attention_probs, v)  # [num_heads, max_len, head_size]

        # 合并多头
        context_layer = context_layer.permute(1, 0, 2).contiguous()  # [max_len, num_heads, head_size]
        context_layer = context_layer.view(-1, hidden_size)  # [max_len, hidden_size]

        # 输出
        attention_output = F.linear(context_layer, attention_output_weight.T, attention_output_bias)
        return attention_output

    def feed_forward(self, x, intermediate_weight, intermediate_bias,
                     output_weight, output_bias):
        # 中间层
        intermediate_output = F.linear(x, intermediate_weight.T, intermediate_bias)
        intermediate_output = F.gelu(intermediate_output)  # 使用GELU激活函数

        # 输出层
        output = F.linear(intermediate_output, output_weight.T, output_bias)
        return output

    def single_transformer_layer_forward(self, x, layer_index=0):
        """单层Transformer前向传播"""
        # 自注意力层
        attention_output = self.self_attention(
            x, self.q_w, self.q_b, self.k_w, self.k_b, self.v_w, self.v_b,
            self.attention_output_weight, self.attention_output_bias,
            self.num_attention_heads, self.hidden_size
        )

        # 残差连接和层归一化
        x = self.layer_norm(x + attention_output, self.attention_layer_norm_w, self.attention_layer_norm_b)

        # 前馈网络
        feed_forward_output = self.feed_forward(
            x, self.intermediate_weight, self.intermediate_bias,
            self.output_weight, self.output_bias
        )

        # 残差连接和层归一化
        x = self.layer_norm(x + feed_forward_output, self.ff_layer_norm_w, self.ff_layer_norm_b)
        return x

    def all_transformer_layer_forward(self, x):
        return self.single_transformer_layer_forward(x)

    def pooler_output_layer(self, x):
        x = F.linear(x, self.pooler_dense_weight.T, self.pooler_dense_bias)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        """完整前向传播"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif isinstance(x, list):
            x = torch.tensor(x)

        # embedding层
        embedding_output = self.embedding_forward(x)

        # transformer层
        sequence_output = self.all_transformer_layer_forward(embedding_output)

        # pooler层（使用[CLS] token）
        pooler_output = self.pooler_output_layer(sequence_output[0])  # 取第一个token ([CLS])

        return sequence_output, pooler_output


# 测试代码
if __name__ == "__main__":
    # 加载预训练模型
    bert = BertModel.from_pretrained("bert-base-chinese", return_dict=False)
    state_dict = bert.state_dict()
    bert.eval()

    # 输入数据
    x = np.array([2450, 15486, 102, 2110])  # 假设的输入序列
    torch_x = torch.LongTensor([x])

    # 原始BERT输出
    with torch.no_grad():
        sequence_output, pooler_output = bert(torch_x)
        print("Original BERT output shapes:", sequence_output.shape, pooler_output.shape)

    # 自己的实现
    diy_bert_torch = DiyBertTorch(state_dict)
    diy_sequence_output, diy_pooler_output = diy_bert_torch.forward(x)

    print("Our implementation output shapes:", diy_sequence_output.shape, diy_pooler_output.shape)

    # 比较结果
    print("\nSequence output difference:", torch.mean(torch.abs(sequence_output[0] - diy_sequence_output)))
    print("Pooler output difference:", torch.mean(torch.abs(pooler_output[0] - diy_pooler_output)))
