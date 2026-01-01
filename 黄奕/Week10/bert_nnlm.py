#coding:utf8
import torch
import torch.nn as nn
import numpy as np
import math, random, os, re
from transformers import BertTokenizer, BertModel


class LanguageModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        super(LanguageModel, self).__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_path,
                                              return_dict=False,
                                              attn_implementation='eager')
        self.classify = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, attention_mask=None):
        if y is not None:
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).unsqueeze(0).expand(x.size(0), -1, -1)
            if x.is_cuda: mask = mask.cuda()
            x, _ = self.bert(x, attention_mask=mask)
            logits = self.classify(x)
            return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            x, _ = self.bert(x, attention_mask=attention_mask)
            logits = self.classify(x)
            return torch.softmax(logits, dim=-1)


def load_corpus(path):
    with open(path, encoding='gbk') as f:
        return ''.join([line.strip() for line in f])

def build_sample(tokenizer, window_size, corpus):
    start = random.randint(0, len(corpus) - window_size - 1)
    window  = corpus[start : start + window_size]
    target  = corpus[start + 1 : start + window_size + 1]
    x = tokenizer.encode(window,  add_special_tokens=False,
                          padding='max_length', truncation=True, max_length=window_size)
    y = tokenizer.encode(target, add_special_tokens=False,
                          padding='max_length', truncation=True, max_length=window_size)
    return x, y

def build_dataset(sample_num, tokenizer, window_size, corpus):
    xs, ys = [], []
    for _ in range(sample_num):
        x, y = build_sample(tokenizer, window_size, corpus)
        xs.append(x); ys.append(y)
    return torch.LongTensor(xs), torch.LongTensor(ys)


def top_k_sampling(prob, k=30, temp=1.0):
    prob, idx = torch.topk(prob, k)
    prob = torch.softmax(prob / temp, dim=-1)
    return idx[torch.multinomial(prob, 1)].item()

@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50):
    model.eval()
    device = next(model.parameters()).device
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    for _ in range(max_len):
        x = torch.LongTensor([ids]).to(device)
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)
        y = model(x, attention_mask=mask)[0, -1]
        nid = top_k_sampling(y, k=30, temp=0.9)
        ids.append(nid)
        if nid == tokenizer.sep_token_id:
            break

    text = tokenizer.decode(ids, skip_special_tokens=True).replace(' ', '')
    return text


def train(corpus_path, save_weight=True):
    epoch_num      = 20
    batch_size     = 128
    train_sample   = 10000
    window_size    = 32
    learning_rate  = 1e-4
    pretrain_path  = r'./bert-base-chinese'

    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    corpus    = load_corpus(corpus_path)
    model     = LanguageModel(768, tokenizer.vocab_size, pretrain_path)
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print('开始训练……')
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for _ in range(train_sample // batch_size):
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y)
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        avg_loss = np.mean(watch_loss)
        ppl = math.exp(avg_loss)
        print(f'第{epoch+1}轮  loss={avg_loss:.4f}  ppl={ppl:.2f}')
        print(generate(model, tokenizer, '让他在半年之前，就不能做出'))
        print(generate(model, tokenizer, '李慕站在山路上，深深的呼吸'))
    if save_weight:
        os.makedirs('model', exist_ok=True)
        save_path = os.path.join('model', os.path.basename(corpus_path).replace('.txt', '.pth'))
        torch.save(model.state_dict(), save_path)
        print('权重已保存至', save_path)


if __name__ == '__main__':
    train(r'corpus.txt',
          save_weight=True)