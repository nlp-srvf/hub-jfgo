# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        max_length = config["input_max_length"]
        # self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        # self.layer = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True, num_layers=num_layers)
        self.bert = BertModel.from_pretrained(config["bert_path"], return_dict=False)
        self.classify = nn.Linear(self.bert.config.hidden_size, self.bert.config.vocab_size)
        self.loss = nn.functional.cross_entropy

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None,mask=None):
        # x = self.embedding(x)  #input shape:(batch_size, sen_len)
        # x, _ = self.layer(x)      #input shape:(batch_size, sen_len, input_dim)
        if mask is not None:
            x, _ = self.bert(x,attention_mask=mask)
        else:
            x, _ = self.bert(x)
        y_pred = self.classify(x) #ouput:(batch_size, sen_len, num_tags) -> (batch_size * sen_len, num_tags)

        if target is not None:
            loss_mask = (~mask[:, 0, :].bool()).float()
            # loss_mask = loss_mask.long()
            # 只在loss_mask为True的位置计算loss
            # y_pred_flat = y_pred[loss_mask].view(-1, y_pred.shape[-1])
            # target_flat = target[loss_mask]
            y_pred_flat = y_pred.masked_fill(loss_mask.unsqueeze(-1) == 0, -1e9)
            target_flat = target.masked_fill(loss_mask == 0, -100)
            y_pred_flat=y_pred_flat.view(-1, y_pred_flat.shape[-1])
            target_flat=target_flat.view(-1)
            return self.loss(y_pred_flat, target_flat)
            #return self.loss(y_pred_flat, target_flat)
            # return self.loss(y_pred.view(-1, y_pred.shape[-1]), target.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)




if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)