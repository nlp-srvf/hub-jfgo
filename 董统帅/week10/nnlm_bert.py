#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer, BertConfig

BERT_PATH = r'D:\Test\llm-cookbook-main\content\week06\bert-base-chinese'
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH)

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        bert_config = BertConfig.from_pretrained(BERT_PATH)
        bert_config.num_hidden_layers = 2
        bert_config.return_dict = False
        # bert_config.output_attentions = True
        bert_config.ignore_mismatched_sizes = True
        self.layer = BertModel.from_pretrained(BERT_PATH, config=bert_config)
        self.layer.config.num_hidden_layers = 2
        self.layer.config.ignore_mismatched_sizes=True
        self.pad_token_id = BERT_TOKENIZER.pad_token_id
        vocab_size = self.layer.config.vocab_size
        hidden_size = self.layer.config.hidden_size
        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def generate_attn_mask(self, x):
        if x.dim() == 1:
            batch_size, seq_len = 1, x.size()[0]
        else:
            batch_size, seq_len = x.size()[:2]
        custom_mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).float() # [batch, seq_len, seq_len]
        custom_mask = custom_mask.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, seq_len, seq_len] 适配 multi-head
        return custom_mask

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        mask_customized = self.generate_attn_mask(x)
        x, _ = self.layer(x, attention_mask=mask_customized)        #output shape:(batch_size, sen_len, hidden_size)
        # x, _, attns = self.layer(x, attention_mask=mask_customized)
        y_pred = self.classify(x)   #output shape:(batch_size, seq_len, vocab_size)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1)) # y (batch_size, seq_len)
        else:
            return torch.softmax(y_pred, dim=-1)

#加载语料
def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

# bert编码会导致长度变长，多出来CLS PAD SEP, 由于SEP的个数不确定，长度无法确定,
# 由于模型输入长度是确定的，需要进行截断处理，实际样本中包含的字、词对应的token会比window_size指定的少
def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]
    x = [BERT_TOKENIZER.vocab.get(ch, BERT_TOKENIZER.vocab['[UNK]']) for ch in window]
    # CLS加在本不是开头的语句前，但是在CLS单个token下预测第一个token是无效的-100
    # x = [BERT_TOKENIZER.cls_token_id] + x
    y = [BERT_TOKENIZER.vocab.get(ch, BERT_TOKENIZER.vocab['[UNK]']) for ch in target]
    # 这里SEP作为最后一个token，会导致所有的预测最后一个token都是SEP
    # y =  y + [BERT_TOKENIZER.sep_token_id]
    return x, y

#建立数据集
#sample_length 输入需要的样本数量。需要多少生成多少
#vocab 词表
#window_size 样本长度
#corpus 语料字符串
def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

#建立模型
def build_model():
    model = LanguageModel()
    return model

#文本生成测试代码
def generate_sentence(openings, model, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过30字则终止迭代
        while pred_char != "[SEP]" and len(openings) <= 30:
            openings += pred_char
            x = [BERT_TOKENIZER.vocab.get(char, BERT_TOKENIZER.vocab['[UNK]']) for char in openings[-window_size:]]
            x = [BERT_TOKENIZER.cls_token_id] + x
            x = torch.LongTensor([x]) # 1, seq_len, hidden_size
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)
            y = y.squeeze()[-2]
            index = sampling_strategy(y)
            pred_char = BERT_TOKENIZER.convert_ids_to_tokens(int(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)


#计算文本ppl
def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))


def train(corpus_path, save_weight=True):
    epoch_num = 10        #训练轮数
    batch_size = 32       #每次训练样本个数
    train_sample = 5000   #每轮训练总共训练的样本总数
    window_size = 10       #样本文本长度
    corpus = load_corpus(corpus_path)     #加载语料
    model = build_model()    #建立模型
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)   #建立优化器
    print("文本词表模型加载完毕，开始训练")
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus) #构建一组训练样本
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()    #梯度归零
            loss = model(x, y.squeeze())   #计算loss
            loss.backward()      #计算梯度
            optim.step()         #更新权重
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print(generate_sentence("让他在半年之前，就不能做出", model, window_size))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, window_size))
    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace("txt", "pth")
        model_path = os.path.join("model", base_name)
        torch.save(model.state_dict(), model_path)
        return

if __name__ == "__main__":
    train("corpus.txt", True)
    # corpus = load_corpus("corpus.txt")     #加载语料
    # x_i,y_i = build_dataset(10, 10, corpus)
    # print(x_i)
    # print(y_i)
