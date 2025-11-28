
import torch
import numpy as np
from torch import nn
from transformers import BertModel, BertConfig
import torch.nn.functional as F

model_path = "bert-base-chinese"
config = BertConfig.from_pretrained(model_path)
config.return_dict = False
config.num_hidden_layers = 12
config.ignore_mismatched_sizes = True

bert_model = BertModel.from_pretrained(model_path, config=config)
#state_dict中不包含dropout,因dropout不包含任何权重
#但是实际上在自注意力的计算后，dropout层是存在的于注意力得分的归一化之后的
bert_state_dict = bert_model.state_dict()
bert_model.eval()

# x = np.array([2450, 15486, 102, 2110])
# torch_x = torch.LongTensor([x])
# sequence_output, pooler_output = bert_model(torch_x)
# print(sequence_output.shape, pooler_output.shape)

x = np.array([[2450, 15486, 102, 2110], [2451, 15485, 102, 2100]])
torch_x = torch.LongTensor(x)
with torch.no_grad():
    sequence_output, pooler_output = bert_model(torch_x)
    print(sequence_output)
    print(pooler_output)

# input
#   │
#   ▼
# [Self-Attention] = [Self-Attention Output] → dense: Linear → dropout → [Add residual & Norm]
#   │
#   ▼
# [FFN] = [Intermediate] (dense: Linear) → GELU → [Output] (dense: Linear)
#   │
#   ▼
# [Add residual & Norm] → output

class DiyBertModel(nn.Module):
    def __init__(self, state_dict):
        super(DiyBertModel, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 12

        embedding_shape = state_dict['embeddings.word_embeddings.weight'].shape
        self.embeddings_layer = nn.Embedding(embedding_shape[0], embedding_shape[1], padding_idx=0)
        self.embeddings_layer.load_state_dict({
            "weight": state_dict['embeddings.word_embeddings.weight']
        })

        embedding_shape = state_dict['embeddings.position_embeddings.weight'].shape
        self.position_embeddings = nn.Embedding(embedding_shape[0], embedding_shape[1], padding_idx=0)
        self.position_embeddings.load_state_dict({
            "weight": state_dict['embeddings.position_embeddings.weight']
        })

        embedding_shape = state_dict['embeddings.token_type_embeddings.weight'].shape
        self.token_type_embeddings = nn.Embedding(embedding_shape[0], embedding_shape[1], padding_idx=0)
        self.token_type_embeddings.load_state_dict({
            "weight": state_dict['embeddings.token_type_embeddings.weight']
        })

        embedding_shape = state_dict['embeddings.LayerNorm.weight'].shape
        self.embeddings_layer_norm = nn.LayerNorm(embedding_shape, bias=True)
        self.embeddings_layer_norm.load_state_dict({
            "weight": state_dict['embeddings.LayerNorm.weight'],
            "bias": state_dict['embeddings.LayerNorm.bias']
        })

        self.transformer_layers = []
        for layer_index in range(self.num_layers):

            linear_shape = state_dict["encoder.layer.%d.attention.self.query.weight" % layer_index].shape
            q_linear = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            q_linear.load_state_dict({
                "weight": state_dict["encoder.layer.%d.attention.self.query.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.attention.self.query.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.attention.self.key.weight" % layer_index].shape
            k_linear = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            k_linear.load_state_dict({
                "weight": state_dict["encoder.layer.%d.attention.self.key.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.attention.self.key.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.attention.self.value.weight" % layer_index].shape
            v_linear = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            v_linear.load_state_dict({
                "weight": state_dict["encoder.layer.%d.attention.self.value.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.attention.self.value.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.attention.output.dense.weight" % layer_index].shape
            atten_output_dense = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            atten_output_dense.load_state_dict({
                "weight": state_dict["encoder.layer.%d.attention.output.dense.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.attention.output.dense.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % layer_index].shape
            atten_output_layer_norm = nn.LayerNorm(linear_shape, bias=True)
            atten_output_layer_norm.load_state_dict({
                "weight": state_dict["encoder.layer.%d.attention.output.LayerNorm.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.attention.output.LayerNorm.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.intermediate.dense.weight" % layer_index].shape
            intermediate_dense = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            intermediate_dense.load_state_dict({
                "weight": state_dict["encoder.layer.%d.intermediate.dense.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.intermediate.dense.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.output.dense.weight" % layer_index].shape
            linear_output_dense = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
            linear_output_dense.load_state_dict({
                "weight": state_dict["encoder.layer.%d.output.dense.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.output.dense.bias" % layer_index]
            })

            linear_shape = state_dict["encoder.layer.%d.output.LayerNorm.weight" % layer_index].shape
            output_layer_norm = nn.LayerNorm(linear_shape, bias=True)
            output_layer_norm.load_state_dict({
                "weight": state_dict["encoder.layer.%d.output.LayerNorm.weight" % layer_index],
                "bias": state_dict["encoder.layer.%d.output.LayerNorm.bias" % layer_index]
            })

            self.transformer_layers.append([q_linear, k_linear, v_linear, atten_output_dense, atten_output_layer_norm, intermediate_dense, linear_output_dense, output_layer_norm])

        linear_shape = state_dict["pooler.dense.weight"].shape
        self.pooler_dense_layer = nn.Linear(linear_shape[1], linear_shape[0], bias=True)
        self.pooler_dense_layer.load_state_dict({
            "weight": state_dict["pooler.dense.weight"],
            "bias": state_dict["pooler.dense.bias"]
        })

    def single_transformer_layer_forward(self, x, layer_index):
        q_linear, k_linear, v_linear, atten_output_dense, \
            atten_output_layer_norm, intermediate_dense, linear_output_dense, \
            output_layer_norm = self.transformer_layers[layer_index]
        #self attention层
        # x.shape = batch_size * max_len * hidden_size
        q_x = q_linear(x)
        k_x = k_linear(x)
        v_x = v_linear(x)
        #计算注意力得分
        batch_size, max_len, hidden_size = x.shape
        dimension_heads = self.hidden_size // self.num_attention_heads
        # qkv_x_head.shape = batch_size, num_attention_heads, max_len, dimension_heads
        q_x_head = q_x.reshape(batch_size, max_len, self.num_attention_heads, dimension_heads).swapaxes(1, 2)
        k_x_head = k_x.reshape(batch_size, max_len, self.num_attention_heads, dimension_heads).swapaxes(1, 2)
        v_x_head = v_x.reshape(batch_size, max_len, self.num_attention_heads, dimension_heads).swapaxes(1, 2)
        # attention_scores.shape = batch_size, num_attention_heads, max_len, max_len
        attention_scores = torch.matmul(q_x_head, k_x_head.transpose(-2, -1))
        attention_scores /= np.sqrt(dimension_heads)
        attention_probs = F.softmax(attention_scores, dim=-1)
        # attention_output.shape = batch_size, num_attention_heads, max_len, dimension_heads
        attention_output = torch.matmul(attention_probs, v_x_head)
        # attention_output.shape = batch_size, max_len, hidden_size
        attention_output = attention_output.swapaxes(1, 2).reshape(batch_size, max_len, hidden_size)
        #归一化, attention_output.shape = batch_size, max_len, hidden_size
        attention_output = atten_output_dense(attention_output)
        #残差连接
        ff_input = x + attention_output
        #归一化, attention_output.shape = batch_size, max_len, hidden_size
        ff_input = atten_output_layer_norm(ff_input)

        #feed forward层
        ff_output = intermediate_dense(ff_input)
        ff_output = F.gelu(ff_output)
        ff_output = linear_output_dense(ff_output)
        #残差连接
        ff_output = ff_input + ff_output
        #归一化, ff_output.shape = batch_size, max_len, hidden_size
        ff_output = output_layer_norm(ff_output)
        return ff_output

    def forward(self, x):
        # x.shape = batch_size, max_len, hidden_size
        word_embedding = self.embeddings_layer(x)

        # position_embedding的输入 [0, 1, 2, 3]
        x_position_embedding = []
        for i in range(len(x)):
            x_position_embedding.append(list(range(len(x[i]))))
        x_position_embedding = torch.LongTensor(x_position_embedding)
        position_embedding = self.position_embeddings(x_position_embedding)

        # token type embedding,暂时只考虑单输入的情况,为[0, 0, 0, 0]
        x_token_type = []
        for i in range(len(x)):
            x_token_type.append([0] * len(x[i]))
        x_token_type = torch.LongTensor(x_token_type)
        token_type_embedding = self.token_type_embeddings(x_token_type)

        embedding = word_embedding + position_embedding + token_type_embedding
        # 加和后有一个归一化层
        transformer_input = self.embeddings_layer_norm(embedding)

        for layer_index in range(self.num_layers):
            transformer_input = self.single_transformer_layer_forward(transformer_input, layer_index)
        pooler_output = self.pooler_dense_layer(transformer_input[:,0])
        pooler_output = F.tanh(pooler_output)
        return transformer_input, pooler_output
print('diy bert model result---------------')
diy_bert_model = DiyBertModel(bert_state_dict)
diy_bert_model.eval()
with torch.no_grad():
    diy_sequence_output, diy_pooler_output = diy_bert_model(torch_x)
    print(diy_sequence_output)
    print(diy_pooler_output)




