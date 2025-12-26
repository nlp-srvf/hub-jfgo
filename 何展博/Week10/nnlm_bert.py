# coding:utf8
import torch
import torch.nn as nn
import numpy as np
import random
import os
from transformers import BertTokenizer, BertModel

"""
改造BERT为Decoder-only架构（CPU/GPU兼容版）
解决：Torch not compiled with CUDA enabled 错误
"""

class BERTasDecoder(nn.Module):
    def __init__(self, config):
        super(BERTasDecoder, self).__init__()
        self.bert = BertModel.from_pretrained(
            config["bert_path"],
            output_hidden_states=False,
            return_dict=False
        )
        self.lm_head = nn.Linear(self.bert.config.hidden_size, config["vocab_size"])
        self.dropout = nn.Dropout(0.1)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    def build_causal_mask(self, seq_len, device):
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        causal_mask = causal_mask.unsqueeze(0)
        return causal_mask

    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Padding掩码 + 因果掩码
        padding_mask = (input_ids != 0).long()
        causal_mask = self.build_causal_mask(seq_len, device).expand(batch_size, -1, -1)
        combined_mask = padding_mask.unsqueeze(-1) * causal_mask

        # 2. BERT前向传播
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=combined_mask
        )
        bert_output = outputs[0]
        bert_output = self.dropout(bert_output)
        logits = self.lm_head(bert_output)

        # 3. 训练/推理分支
        if labels is not None:
            logits = logits[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            return self.loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
        else:
            return torch.softmax(logits[:, -1, :], dim=-1)

# 加载语料（UTF-8编码）
def load_corpus(path):
    corpus = ""
    with open(path, encoding="utf8", errors='ignore') as f:
        for line in f:
            corpus += line.strip()
    return corpus

# 构建自回归训练样本
def build_sample(tokenizer, seq_len, corpus):
    max_start = len(corpus) - (seq_len + 1)
    if max_start <= 0:
        max_start = 0
    start = random.randint(0, max_start)
    end = start + seq_len + 1
    text = corpus[start:end]

    tokens = tokenizer.tokenize(text)
    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    if len(input_ids) < seq_len + 1:
        input_ids += [0] * (seq_len + 1 - len(input_ids))
    else:
        input_ids = input_ids[:seq_len + 1]

    input_seq = input_ids[:-1]
    label_seq = input_ids[1:]
    return input_seq, label_seq

# 构建数据集
def build_dataset(sample_num, tokenizer, seq_len, corpus):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_num):
        x, y = build_sample(tokenizer, seq_len, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

# 自回归文本生成（CPU/GPU兼容）
def generate_text(start_text, model, tokenizer, seq_len, max_gen_len=50, temperature=0.8):
    model.eval()
    with torch.no_grad():
        # 自动获取设备（和模型一致）
        device = next(model.parameters()).device
        # 初始输入处理
        input_ids = tokenizer.encode(start_text, add_special_tokens=False)
        generated = start_text

        # 自回归生成
        for _ in range(max_gen_len):
            # 填充/截断
            if len(input_ids) < seq_len:
                input_ids = [0] * (seq_len - len(input_ids)) + input_ids
            else:
                input_ids = input_ids[-seq_len:]

            # 转张量（放到模型相同设备）
            input_tensor = torch.LongTensor([input_ids]).to(device)

            # 模型预测
            probs = model(input_tensor) / temperature
            probs = torch.softmax(probs, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).item()

            # ID转字符
            next_token = tokenizer.convert_ids_to_tokens([next_token_id])[0]
            if next_token.startswith("##"):
                next_token = next_token[2:]
            elif next_token == "[UNK]":
                next_token = "。"
            elif next_token_id == 0:
                break

            generated += next_token
            input_ids.append(next_token_id)

        return generated

# 训练函数（CPU/GPU兼容）
def train(corpus_path, save_model=True):
    # 配置参数
    config = {
        "bert_path": r"D:\bert\google-bert\bert-base-chinese",
        "vocab_size": 30522,
        "epoch_num": 5,  # 减少轮数，CPU训练更快
        "batch_size": 8, # 减小批次，CPU内存更友好
        "train_sample_num": 10000, # 减少样本数，加快训练
        "seq_len": 16,
        "lr": 2e-5
    }

    # ========== 核心改动1：自动判断设备 ==========
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备：{device}")

    # 加载分词器和语料
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    corpus = load_corpus(corpus_path)
    print(f"语料加载完成，总长度：{len(corpus)} 字符")

    # 初始化模型（放到指定设备）
    model = BERTasDecoder(config).to(device)
    print("BERT改造为Decoder-only模型完成")

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)

    # 训练主循环
    for epoch in range(config["epoch_num"]):
        model.train()
        total_loss = 0.0
        batch_num = int(config["train_sample_num"] / config["batch_size"])

        for batch_idx in range(batch_num):
            # 生成批次数据
            x, y = build_dataset(config["batch_size"], tokenizer, config["seq_len"], corpus)
            # ========== 核心改动2：张量放到指定设备 ==========
            x, y = x.to(device), y.to(device)

            # 梯度清零 + 训练
            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # 打印批次损失
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{config['epoch_num']} | Batch {batch_idx+1}/{batch_num} | Loss: {loss.item():.4f}")

        # 每轮训练后打印平均损失 + 生成示例
        avg_loss = total_loss / batch_num
        print(f"\n===== Epoch {epoch+1} 训练完成 | 平均损失：{avg_loss:.4f} =====")
        # 生成测试文本
        test_starts = [
            "清晨的阳光洒在",
            "他走在乡间的小路上，"
        ]
        for idx, start_text in enumerate(test_starts):
            gen_text = generate_text(start_text, model, tokenizer, config["seq_len"])
            print(f"生成示例{idx+1}：{gen_text}")

    # 保存模型
    if save_model:
        os.makedirs("bert_decoder_models", exist_ok=True)
        model_path = os.path.join("bert_decoder_models", "bert_decoder_text_gen.pth")
        # 保存时剥离设备信息，兼容CPU/GPU
        torch.save(model.state_dict(), model_path)
        print(f"\n模型已保存至：{model_path}")

if __name__ == "__main__":
    # 替换为你的语料文件路径
    train("corpus.txt", save_model=True)