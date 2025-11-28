#coding:utf8

"""
title：基于pytorch实现transform部分功能
time: 2025-11-27
version: 0.1
detail: 目前暂时无法运行训练，只是大致上描述torch实现transform的结构
"""

import torch
import torch.nn as nn
import math
import numpy as np
from transformers import BertModel


#softmax归一化
def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=-1, keepdims=True)

#gelu激活函数
def gelu(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * np.power(x, 3))))


class TorchBert(nn.Module):
    def __init__(self, state_dict,maxlen):
        super(TorchBert, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 1        #注意这里的层数要跟预训练config.json文件中的模型层数一致
        self.load_weights(state_dict)
        #embedding层借用bert自带的
        # self.token_embedding = nn.Embedding(len(vocab) + 1, input_dim, padding_idx=0)
        # self.segment_embedding = nn.Embedding(2, input_dim)
        # self.position_embedding = nn.Embedding(512, input_dim)
        self.layer_q = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_k = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_v = nn.Linear(self.hidden_size, self.hidden_size)
        self.layer_o = nn.Linear(self.hidden_size, self.hidden_size)
        self.feed_forward_1 = nn.Linear(self.hidden_size, 4*self.hidden_size)
        self.feed_forward_2 = nn.Linear(4*self.hidden_size, self.hidden_size)

        self.loss = nn.functional.mse_loss

    def load_weights(self, state_dict):
        #embedding部分
        self.word_embeddings = state_dict["embeddings.word_embeddings.weight"].numpy()
        self.position_embeddings = state_dict["embeddings.position_embeddings.weight"].numpy()
        self.token_type_embeddings = state_dict["embeddings.token_type_embeddings.weight"].numpy()
        self.embeddings_layer_norm_weight = state_dict["embeddings.LayerNorm.weight"].numpy()
        self.embeddings_layer_norm_bias = state_dict["embeddings.LayerNorm.bias"].numpy()
        self.transformer_weights = []
        #transformer部分，有多层
        for i in range(self.num_layers):
            q_w = state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy()
            q_b = state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy()
            k_w = state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy()
            k_b = state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy()
            v_w = state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy()
            v_b = state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy()
            attention_output_weight = state_dict["encoder.layer.%d.attention.output.dense.weight" % i].numpy()
            attention_output_bias = state_dict["encoder.layer.%d.attention.output.dense.bias" % i].numpy()
            attention_layer_norm_w = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % i].numpy()
            attention_layer_norm_b = state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % i].numpy()
            intermediate_weight = state_dict["encoder.layer.%d.intermediate.dense.weight" % i].numpy()
            intermediate_bias = state_dict["encoder.layer.%d.intermediate.dense.bias" % i].numpy()
            output_weight = state_dict["encoder.layer.%d.output.dense.weight" % i].numpy()
            output_bias = state_dict["encoder.layer.%d.output.dense.bias" % i].numpy()
            ff_layer_norm_w = state_dict["encoder.layer.%d.output.LayerNorm.weight" % i].numpy()
            ff_layer_norm_b = state_dict["encoder.layer.%d.output.LayerNorm.bias" % i].numpy()
            self.transformer_weights.append([q_w, q_b, k_w, k_b, v_w, v_b, attention_output_weight, attention_output_bias,
                                             attention_layer_norm_w, attention_layer_norm_b, intermediate_weight, intermediate_bias,
                                             output_weight, output_bias, ff_layer_norm_w, ff_layer_norm_b])
        #pooler层
        self.pooler_dense_weight = state_dict["pooler.dense.weight"].numpy()
        self.pooler_dense_bias = state_dict["pooler.dense.bias"].numpy()


    #bert embedding，使用3层叠加，在经过一个Layer norm层
    def embedding_forward(self, x):
        # x.shape = [max_len]
        we = self.get_embedding(self.word_embeddings, x)  # shpae: [ , hidden_size]
        # position embeding的输入 [0, 1, 2, 3]
        pe = self.get_embedding(self.position_embeddings, np.array(list(range(len(x)))))  # shpae: [max_len, hidden_size]
        # token type embedding,单输入的情况下为[0, 0, 0, 0]
        te = self.get_embedding(self.token_type_embeddings, np.array([0] * len(x)))  # shpae: [max_len, hidden_size]
        embedding = we + pe + te
        # 加和后有一个归一化层
        embedding = self.layer_norm(embedding, self.embeddings_layer_norm_weight, self.embeddings_layer_norm_bias)  # shpae: [max_len, hidden_size]
        return embedding

    #embedding层实际上相当于按index索引，或理解为onehot输入乘以embedding矩阵
    def get_embedding(self, embedding_matrix, x):
        return np.array([embedding_matrix[index] for index in x])



    #执行全部的transformer层计算
    def all_transformer_layer_forward(self, x):
        for i in range(self.num_layers):
            x = self.single_transformer_layer_forward(x, i)
        return x

    #执行单层transformer层计算
    def single_transformer_layer_forward(self, x, layer_index):
        weights = self.transformer_weights[layer_index]

        #取出该层的参数，在实际中，这些参数都是随机初始化，之后进行预训练
        q_w, q_b, \
        k_w, k_b, \
        v_w, v_b, \
        attention_output_weight, attention_output_bias, \
        attention_layer_norm_w, attention_layer_norm_b, \
        intermediate_weight, intermediate_bias, \
        output_weight, output_bias, \
        ff_layer_norm_w, ff_layer_norm_b = weights
        #self attention层

        attention_output = self.self_attention(x,
                                attention_output_weight, attention_output_bias,
                                self.num_attention_heads,
                                self.hidden_size)
        #bn层，并使用了残差机制
        #gelu 归一化
        x = self.layer_norm(x + attention_output, attention_layer_norm_w, attention_layer_norm_b)
        #feed forward层
        feed_forward_x = self.feed_forward(x,
                              intermediate_weight, intermediate_bias,
                              output_weight, output_bias)
        #bn层，并使用了残差机制
        x = self.layer_norm(x + feed_forward_x, ff_layer_norm_w, ff_layer_norm_b)
        return x

    # self attention的计算
    def self_attention(self,
                       x,
                       attention_output_weight,
                       attention_output_bias,
                       num_attention_heads,
                       hidden_size):
        # x.shape = max_len * hidden_size
        q = self.layer_q(x)    # shape: [max_len, hidden_size]
        k = self.layer_k(x)    # shape: [max_len, hidden_size]
        v = self.layer_v(x)    # shape: [max_len, hidden_size]
        kt = k.transpose(0, 1)

        # attention = torch.bmm( softmax( torch.bmm( xq, xk)/dk), xv)

        attention_head_size = int(hidden_size / num_attention_heads)

        # q.shape = num_attention_heads, max_len, attention_head_size
        q = self.transpose_for_scores(q, attention_head_size, num_attention_heads)
        # k.shape = num_attention_heads, max_len, attention_head_size
        k = self.transpose_for_scores(k, attention_head_size, num_attention_heads)
        # v.shape = num_attention_heads, max_len, attention_head_size
        v = self.transpose_for_scores(v, attention_head_size, num_attention_heads)
        # qk.shape = num_attention_heads, max_len, max_len
        qk = torch.matmul(q ,kt)
        qk /= np.sqrt(attention_head_size)
        qk = softmax(qk)
        # qkv.shape = num_attention_heads, max_len, attention_head_size
        qkv = torch.matmul(qk, v)
        # qkv.shape = max_len, hidden_size
        # qkv = qkv.swapaxes(0, 1).reshape(-1, hidden_size)
        qkv = qkv.transpose(0, 1).reshape(-1, hidden_size)
        # attention.shape = max_len, hidden_size
        attention = self.layer_o(qkv)
        return attention

    #多头机制
    def transpose_for_scores(self, x, attention_head_size, num_attention_heads):
        # hidden_size = 768  num_attent_heads = 12 attention_head_size = 64
        max_len, hidden_size = x.size()
        x = x.reshape(max_len, num_attention_heads, attention_head_size)
        x = x.swapaxes(1, 0)  # output shape = [num_attention_heads, max_len, attention_head_size]
        return x

    #前馈网络的计算
    def feed_forward(self,
                     x,
                     intermediate_weight,  # intermediate_size, hidden_size
                     intermediate_bias,  # intermediate_size
                     output_weight,  # hidden_size, intermediate_size
                     output_bias,  # hidden_size
                     ):
        # output shpae: [max_len, intermediate_size]
        x = self.feed_forward_1(x)
        x = gelu(x)
        # output shpae: [max_len, hidden_size]
        x = self.feed_forward_2(x)
        return x

    #归一化层
    def layer_norm(self, x, w, b):
        x = (x - np.mean(x, axis=1, keepdims=True)) / np.std(x, axis=1, keepdims=True)
        x = x * w + b
        return x

    # #链接[cls] token的输出层
    # def pooler_output_layer(self, x):
    #     x = np.dot(x, self.pooler_dense_weight.T) + self.pooler_dense_bias
    #     x = np.tanh(x)
    #     return x

    #最终输出
    def forward(self, x , y=None ):
        x = self.embedding_forward(x)
        sequence_output = self.all_transformer_layer_forward(x)
        # pooler_output = self.pooler_output_layer(sequence_output[0])
        y_pred = self.activation(sequence_output)
        if y is not None:
            return self.loss(y_pred, y.squeeze())
        else:
            return y_pred

def main():
    pass
    # bert = BertModel.from_pretrained(r"", return_dict=False)
    # state_dict = bert.state_dict()
    # bert.eval()
    # x = np.array([2450, 15486, 102, 2110])  # 假想成4个字的句子
    # torch_x = torch.LongTensor([x])  # pytorch形式输入
    # seqence_output, pooler_output = bert(torch_x)
    # print(seqence_output.shape, pooler_output.shape)
    # # print(seqence_output, pooler_output)
    # print(bert.state_dict().keys())  # 查看所有的权值矩阵名称
    #
    #
    # # 自制
    # db = TorchBert(state_dict)
    # diy_sequence_output, diy_pooler_output = db.forward(x)
    # # torch
    # torch_sequence_output, torch_pooler_output = bert(torch_x)
    #
    # print(diy_sequence_output)
    # print(torch_sequence_output)

if __name__ == '__main__':
    main()
