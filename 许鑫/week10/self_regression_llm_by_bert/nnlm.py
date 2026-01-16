# coding:utf8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from transformers import BertModel, BertTokenizer

"""
使用 BERT 作为【伪自回归语言模型】进行训练
仅预测最后一个 token
"""

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BERT_PATH = "../../bert-base-chinese"


class LanguageModel(nn.Module):
    def __init__(self, vocab_size, pad_id):
        super().__init__()
        self.pad_id = pad_id
        self.bert = BertModel.from_pretrained(
            BERT_PATH,
            return_dict=False
        )
        self.classifier = nn.Linear(768, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        input_ids: (B, L)
        attention_mask: (B, L)
        labels: (B, L)
        """
        hidden_states, _ = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )  # (B, L, 768)

        logits = self.classifier(hidden_states)  # (B, L, vocab)

        # 训练：只预测最后一个 token
        if labels is not None:

            loss = F.cross_entropy(
                logits[:, -1, :],
                labels[:, -1],
                ignore_index=self.pad_id
            )
            return loss
        else:
            return torch.softmax(logits[:, -1, :], dim=-1)


def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(tokenizer, corpus, window_size):
    """
    从语料中截取一段：
    输入：前 window_size 个 token
    标签：右移一位
    """
    start = random.randint(0, len(corpus) - window_size - 1)
    text = corpus[start:start + window_size + 1]

    tokens = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors="pt"
    )["input_ids"][0]

    input_ids = tokens[:-1]
    labels = tokens[1:]

    return input_ids, labels


def build_dataset(tokenizer, corpus, batch_size, window_size):
    xs, ys = [], []
    for _ in range(batch_size):
        x, y = build_sample(tokenizer, corpus, window_size)
        xs.append(x)
        ys.append(y)
    pad_id = tokenizer.pad_token_id
    xs = nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad_id)
    ys = nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=pad_id)

    attention_mask = (xs != 0).long()
    return xs.to(DEVICE), attention_mask.to(DEVICE), ys.to(DEVICE)


def sampling_strategy(prob, tokenizer, temperature=1.2):
    prob = prob.clone()

    # 禁止非法 token
    prob[tokenizer.pad_token_id] = 0
    prob[tokenizer.unk_token_id] = 0

    # 防止全 0
    if prob.sum() <= 0:
        prob = torch.ones_like(prob)
        prob[tokenizer.pad_token_id] = 0
        prob[tokenizer.unk_token_id] = 0

    # greedy or sampling
    if random.random() > 0.1:
        return int(torch.argmax(prob))
    else:
        prob = torch.log(prob + 1e-9) / temperature
        prob = torch.softmax(prob, dim=-1)
        prob = prob.cpu().numpy()
        prob = prob / prob.sum()
        return np.random.choice(len(prob), p=prob)


def generate_sentence(opening, model, tokenizer, window_size=20, max_len=30):
    model.eval()
    text = opening

    with torch.no_grad():
        for _ in range(max_len):
            tokens = tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False
            )
            input_ids = tokens["input_ids"][:, -window_size:].to(DEVICE)
            attention_mask = (input_ids != 0).long()

            prob = model(input_ids, attention_mask)
            next_id = sampling_strategy(prob[0], tokenizer)
            next_char = tokenizer.decode([next_id])

            text += next_char
            if next_char == "\n":
                break

    return text


def train(corpus_path):
    epoch_num = 10
    batch_size = 64
    train_steps = 1000
    window_size = 80
    lr = 1e-5

    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    corpus = load_corpus(corpus_path)

    model = LanguageModel(tokenizer.vocab_size, tokenizer.pad_token_id).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    print("模型加载完成，开始训练")

    for epoch in range(epoch_num):
        model.train()
        losses = []

        for step in range(train_steps):
            x, mask, y = build_dataset(
                tokenizer, corpus, batch_size, window_size
            )

            optimizer.zero_grad()
            loss = model(x, mask, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"\nEpoch {epoch + 1} | loss: {np.mean(losses):.4f}")
        print(generate_sentence("让他在半年之前，就不能做出", model, tokenizer))
        print(generate_sentence("李慕站在山路上，深深的呼吸", model, tokenizer))


if __name__ == "__main__":
    train("corpus.txt")
