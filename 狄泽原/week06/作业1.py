import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel


class DiyBertPytorch(nn.Module):
    def __init__(self, pretrained_bert_path):
        super(DiyBertPytorch, self).__init__()
        self.num_attention_heads = 12
        self.hidden_size = 768
        self.num_layers = 6
        self.intermediate_size = 3072

        pretrained_bert = BertModel.from_pretrained(pretrained_bert_path , return_dict=False)
        self._init_layers()
        self.load_pretrained(pretrained_bert.state_dict())

    def _init_layers(self):
        self.word_embeddings = nn.Embedding.from_pretrained(torch.zeros(21128,self.hidden_size))
        self.position_embeddings = nn.Embedding.from_pretrained(torch.zeros(512,self.hidden_size))
        self.token_type_embeddings = nn.Embedding.from_pretrained(torch.zeros(2,self.hidden_size))

        self.embedding_layer_norm = nn.LayerNorm(self.hidden_size,eps=1e-12)

        self.transformer_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            attention_layer = nn.ModuleDict({
                "Q":nn.Linear(self.hidden_size,self.hidden_size),
                "K":nn.Linear(self.hidden_size,self.hidden_size),
                "V":nn.Linear(self.hidden_size,self.hidden_size),
                "output":nn.Linear(self.hidden_size,self.hidden_size),
                "layer_norm":nn.LayerNorm(self.hidden_size,eps=1e-12)
            })
            ffn_layer = nn.ModuleDict({
                "intermediate":nn.Linear(self.hidden_size,self.intermediate_size),
                "output":nn.Linear(self.intermediate_size,self.hidden_size),
                "layer_norm":nn.LayerNorm(self.hidden_size,eps=1e-12)
            })

            transformer_layer = nn.ModuleDict({
                "attention":attention_layer,
                "ffn":ffn_layer,
            })
            self.transformer_layers.append(transformer_layer)
        self.pooler = nn.Linear(self.hidden_size,self.hidden_size)


    def load_pretrained(self, state_dict):
        self.word_embeddings.weight.data = torch.tensor(state_dict["embeddings.word_embeddings.weight"].numpy())
        self.position_embeddings.weight.data = torch.tensor(state_dict["embeddings.position_embeddings.weight"].numpy())
        self.token_type_embeddings.weight.data = torch.tensor(state_dict["embeddings.token_type_embeddings.weight"].numpy())
        self.embedding_layer_norm.weight.data = torch.tensor(state_dict["embeddings.LayerNorm.weight"].numpy())
        self.embedding_layer_norm.bias.data = torch.tensor(state_dict["embeddings.LayerNorm.bias"].numpy())

        for i ,layer in enumerate(self.transformer_layers):
            layer["attention"]["Q"].weight.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.query.weight" % i].numpy())
            layer["attention"]["Q"].bias.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.query.bias" % i].numpy())
            layer["attention"]["K"].weight.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.key.weight" % i].numpy())
            layer["attention"]["K"].bias.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.key.bias" % i].numpy())
            layer["attention"]["V"].weight.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.value.weight" % i].numpy())
            layer["attention"]["V"].bias.data = torch.tensor(state_dict["encoder.layer.%d.attention.self.value.bias" % i].numpy())

            layer["attention"]["output"].weight.data = torch.tensor(state_dict[f"encoder.layer.{i}.attention.output.dense.weight"].numpy())
            layer["attention"]["output"].bias.data = torch.tensor(state_dict[f"encoder.layer.{i}.attention.output.dense.bias"].numpy())
            layer["attention"]["layer_norm"].weight.data = torch.tensor(state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.weight"].numpy())
            layer["attention"]["layer_norm"].bias.data = torch.tensor(state_dict[f"encoder.layer.{i}.attention.output.LayerNorm.bias"].numpy())

            # FFN层权重
            layer["ffn"]["intermediate"].weight.data = torch.tensor(state_dict[f"encoder.layer.{i}.intermediate.dense.weight"].numpy(), dtype=torch.float32 )
            layer["ffn"]["intermediate"].bias.data = torch.tensor(state_dict[f"encoder.layer.{i}.intermediate.dense.bias"].numpy(), dtype=torch.float32)
            layer["ffn"]["output"].weight.data = torch.tensor(state_dict[f"encoder.layer.{i}.output.dense.weight"].numpy(), dtype=torch.float32)
            layer["ffn"]["output"].bias.data = torch.tensor(state_dict[f"encoder.layer.{i}.output.dense.bias"].numpy(), dtype=torch.float32)
            layer["ffn"]["layer_norm"].weight.data = torch.tensor( state_dict[f"encoder.layer.{i}.output.LayerNorm.weight"].numpy(), dtype=torch.float32)
            layer["ffn"]["layer_norm"].bias.data = torch.tensor(state_dict[f"encoder.layer.{i}.output.LayerNorm.bias"].numpy(), dtype=torch.float32)

        # 3. Pooler层权重
        self.pooler.weight.data = torch.tensor(state_dict["pooler.dense.weight"].numpy(), dtype=torch.float32 )
        self.pooler.bias.data = torch.tensor(state_dict["pooler.dense.bias"].numpy(), dtype=torch.float32)


    def embedding_forward(self, x):
        batch_size ,max_len = x.shape

        we = self.word_embeddings(x)
        position_ids = torch.arange(max_len, device=x.device).unsqueeze(0).repeat(batch_size, 1)
        pe = self.position_embeddings(position_ids)
        token_type_ids = torch.zeros_like(x, device=x.device)
        te = self.token_type_embeddings(token_type_ids)
        embedding = we + pe + te
        embedding = self.embedding_layer_norm(embedding)
        return embedding

    def transpose_for_scores(self, x):
        batch_size, max_len, hidden_size = x.shape
        attention_head_size = self.hidden_size // self.num_attention_heads

        x = x.view(batch_size, max_len, self.num_attention_heads, attention_head_size)
        return x.permute(0, 2, 1, 3)

    def self_attention_forward(self, x, attention_layer):
        batch_size, max_len, hidden_size = x.shape

        q = attention_layer["Q"](x)
        k = attention_layer["K"](x)
        v = attention_layer["V"](x)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        attention_scores = torch.matmul(q, k.transpose(-1, -2))
        attention_scores = attention_scores / torch.sqrt(torch.tensor(
            self.hidden_size // self.num_attention_heads, dtype=torch.float32, device=x.device
        ))
        attention_weights = F.softmax(attention_scores, dim=-1)

        attention_output = torch.matmul(attention_weights, v)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, max_len, hidden_size)
        attention_output = attention_layer["output"](attention_output)

        return attention_output

    def ffn_forward(self, x, ffn_layer):

        x = ffn_layer["intermediate"](x)
        x = F.gelu(x)
        x = ffn_layer["output"](x)
        return x

    def single_transformer_layer(self, x, transformer_layer):
        attention_output = self.self_attention_forward(x, transformer_layer["attention"])
        x = transformer_layer["attention"]["layer_norm"](x + attention_output)

        ffn_output = self.ffn_forward(x, transformer_layer["ffn"])
        x = transformer_layer["ffn"]["layer_norm"](x + ffn_output)
        return x

    def pooler_forward(self, x):
        cls_token = x[:, 0, :]
        pooler_output = self.pooler(cls_token)
        pooler_output = torch.tanh(pooler_output)
        return pooler_output

    def forward(self, x):
        x = self.embedding_forward(x)

        for layer in self.transformer_layers:
            x = self.single_transformer_layer(x, layer)

        sequence_output = x
        pooler_output = self.pooler_forward(x)

        return sequence_output, pooler_output


if __name__ == "__main__":
    # 配置
    pretrained_bert_path = r"F:\llm\六\bert-base-chinese\bert-base-chinese"
    batch_size = 2
    max_len = 4

    diy_bert = DiyBertPytorch(pretrained_bert_path)
    diy_bert.eval()

    native_bert = BertModel.from_pretrained(pretrained_bert_path, return_dict=False)
    native_bert.eval()

    test_input = torch.tensor([[2450, 15486, 102, 2110], [101, 2450, 15486, 102]], dtype=torch.long)

    with torch.no_grad():
        diy_seq, diy_pool = diy_bert(test_input)
        native_seq, native_pool = native_bert(test_input)

    print(diy_seq)
    print(native_seq)


