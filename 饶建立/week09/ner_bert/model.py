# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from transformers import BertModel
from torchcrf import CRF

"""
建立网络模型结构 - BERT版本
使用BERT作为编码器进行序列标注任务
"""

class BertNERModel(nn.Module):
    def __init__(self, config):
        super(BertNERModel, self).__init__()
        self.config = config
        class_num = config["class_num"]
        self.use_crf = config["use_crf"]
        
        # 加载BERT预训练模型
        self.bert = BertModel.from_pretrained(config["bert_path"])
        hidden_size = self.bert.config.hidden_size  # 768 for bert-base
        
        # 分类层：将BERT输出映射到标签空间
        self.classify = nn.Linear(hidden_size, class_num)
        
        # CRF层
        self.crf_layer = CRF(class_num, batch_first=True)
        
        # 交叉熵损失（不使用CRF时）
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, input_ids, attention_mask=None, target=None):
        """
        前向传播
        :param input_ids: 输入的token ids, shape: (batch_size, seq_len)
        :param attention_mask: 注意力掩码, shape: (batch_size, seq_len)
        :param target: 目标标签, shape: (batch_size, seq_len)
        :return: 损失值或预测结果
        """
        # BERT编码
        # outputs.last_hidden_state shape: (batch_size, seq_len, hidden_size)
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # 分类
        # predict shape: (batch_size, seq_len, class_num)
        predict = self.classify(sequence_output)
        
        if target is not None:
            # 训练模式：计算损失
            if self.use_crf:
                # CRF需要mask来忽略padding部分
                # 使用target > -1作为mask（-1是padding的标签）
                mask = target.gt(-1)
                return -self.crf_layer(predict, target, mask, reduction="mean")
            else:
                # 使用交叉熵损失
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            # 推理模式：返回预测结果
            if self.use_crf:
                return self.crf_layer.decode(predict)
            else:
                return predict


def choose_optimizer(config, model):
    """
    选择优化器
    对于BERT，通常使用分层学习率：BERT层使用较小学习率，分类层使用较大学习率
    """
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    
    # 分层学习率
    bert_params = list(model.bert.parameters())
    other_params = list(model.classify.parameters())
    if config["use_crf"]:
        other_params += list(model.crf_layer.parameters())
    
    # BERT层使用较小学习率，其他层使用较大学习率
    param_groups = [
        {"params": bert_params, "lr": learning_rate},
        {"params": other_params, "lr": learning_rate * 10}  # 分类层学习率更大
    ]
    
    if optimizer == "adam":
        return Adam(param_groups)
    elif optimizer == "sgd":
        return SGD(param_groups)


if __name__ == "__main__":
    from config import Config
    model = BertNERModel(Config)
    print(model)
    
    # 测试前向传播
    batch_size = 2
    seq_len = 10
    input_ids = torch.randint(0, 21128, (batch_size, seq_len))  # BERT vocab size
    attention_mask = torch.ones(batch_size, seq_len)
    target = torch.randint(0, 9, (batch_size, seq_len))
    
    # 计算损失
    loss = model(input_ids, attention_mask, target)
    print(f"Loss: {loss}")
    
    # 预测
    pred = model(input_ids, attention_mask)
    print(f"Prediction shape: {len(pred)}, {len(pred[0])}")

