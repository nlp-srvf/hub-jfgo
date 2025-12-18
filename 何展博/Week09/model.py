# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

"""建立网络模型结构"""
class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        self.classify = nn.Linear(self.bert.config.hidden_size, config["class_num"])
        self.crf_layer = CRF(config["class_num"], batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        # x is input_ids
        outputs = self.bert(x)
        sequence_output = outputs[0]
        predict = self.classify(sequence_output)
        if target is not None:
            if self.use_crf:
                mask = target.gt(-1)
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
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


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)