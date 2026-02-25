# -*- coding: utf-8 -*-
from config import Config
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel, BertTokenizer
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        vocab_size = config["vocab_size"] + 1
        max_length = config["max_length"]
        class_num = config["class_num"]
        num_layers = config["num_layers"]
        # 原始代码
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)

        self.bert = BertModel.from_pretrained(r"D:\pretrain_models\bert-base-chinese", return_dict=False)
        hidden_size = self.bert.config.hidden_size  # BERT 的隐藏层大小，通常是 768
        self.classify = nn.Linear(hidden_size , class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    # x: input_ids (batch_size, sen_len)
    # attention_mask (batch_size, sen_len)
    # token_type_ids (batch_size, sen_len)
    def forward(self, input_ids, attention_mask=None, token_type_ids=None,labels=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, hidden_size * 2)
        sequence_output, pooler_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        predict = self.classify(sequence_output) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if labels is not None:
            if self.use_crf:
                mask = labels.gt(-1)
                return - self.crf_layer(predict, labels, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), labels.view(-1))
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer:{optimizer}")


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
