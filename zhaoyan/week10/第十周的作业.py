# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
from transformers import BertTokenizer, BertModel, BertConfig

"""
基于BERT的语言模型，使用因果掩码实现自回归生成
"""


class LanguageModel(nn.Module):
    def __init__(self, config):
        super(LanguageModel, self).__init__()
        self.config = config

        # 加载BERT tokenizer和模型
        self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab_size = len(self.tokenizer)

        # 初始化BERT模型
        bert_config = BertConfig.from_pretrained(
            config["pretrain_model_path"],
            vocab_size=self.vocab_size,
            is_decoder=True,  # 设置为decoder模式
            add_cross_attention=False
        )
        self.bert = BertModel(bert_config)

        # 分类头
        self.classify = nn.Linear(bert_config.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def create_causal_mask(self, seq_len, device):
        """
        创建因果掩码（Causal Mask）
        用于自回归语言模型，防止模型看到未来的token
        形状: [seq_len, seq_len]
        下三角为1（包括对角线），上三角为0
        """
        # 创建一个下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask

    def forward(self, input_ids, attention_mask=None, labels=None):
        """
        前向传播
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] (padding mask)
        labels: [batch_size, seq_len] (用于计算loss)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 1. 创建padding mask（如果有）
        if attention_mask is None:
            attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # 2. 创建因果掩码
        causal_mask = self.create_causal_mask(seq_len, device)  # [seq_len, seq_len]
        # 3. 扩展维度以匹配batch
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, seq_len]
        causal_mask = causal_mask.expand(batch_size, -1, -1, -1)  # [batch_size, 1, seq_len, seq_len]

        # 4. 将padding mask转换为attention mask的格式
        # BERT期望的attention mask形状: [batch_size, 1, 1, seq_len] 或 [batch_size, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, seq_len]

        # 5. 合并padding mask和causal mask
        # 只有padding位置为0且causal允许的位置为1才有效 [batch_size, 1, seq_len, seq_len]
        combined_mask = extended_attention_mask * causal_mask

        # 6.
        # BERT使用加法形式的mask，所以需要转换
        combined_mask = (1.0 - combined_mask) * -10000

        # 既然设置了 is_decoder=True，BERT会自动处理因果掩码
        # 我们只需要提供padding mask，让BERT自己处理
        # 7. 通过BERT模型
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # 8. 获取序列输出
        sequence_output = outputs[0]  # [batch_size, seq_len, hidden_size]

        # 9. 分类层
        logits = self.classify(sequence_output)  # [batch_size, seq_len, vocab_size]

        # input_ids = [CLS, 今, 天, 天, 气, 很, SEP]
        # labels = [今, 天, 天, 气, 很, SEP, PAD]
        #
        # logits = model(input_ids)
        # [CLS, 今, 天, 天, 气, 很, SEP]
        #
        # shift_logits = logits[:, :-1, :]  # 取前6个位置的预测（去掉最后一个）
        # [CLS, 今, 天, 天, 气, 很]
        #
        #
        # shift_labels = labels[:, :-1]  # 前6个位置的标签（去掉第一个）
        # [今, 天, 天, 气, 很, SEP, ]

        # 10. 计算loss或返回预测
        if labels is not None:
            # 计算loss，忽略padding位置
            shift_logits = logits[:, :-1, :].contiguous()  # 移除最后一个token的预测
            shift_labels = labels[:, :-1] .contiguous()  # 移除第一个token的标签
            loss_mask = attention_mask[:, 1:].contiguous()  # 对应的mask

            loss = self.loss(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            # 可选：根据mask加权
            # loss = (loss.view(shift_labels.shape) * loss_mask).sum() / loss_mask.sum()
            return loss
        else:
            return torch.softmax(logits, dim=-1)


def load_corpus(path):
    """加载语料"""
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus


def build_sample(tokenizer, window_size, corpus):
    """构建训练样本"""
    # 随机选择起始位置
    start = random.randint(0, len(corpus) - window_size - 1)
    end = start + window_size
    sentence = corpus[start:end]

    # 使用tokenizer编码，添加特殊token
    encoding = tokenizer(
        sentence,
        max_length=tokenizer.model_max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # input_ids已经包含了[CLS]和[SEP]
    input_ids = encoding['input_ids'].squeeze(0)  # [seq_len]

    # 标签是input_ids向右移动一位（用于预测下一个token）
    labels = input_ids.clone()
    # 将input_ids向右移动一位，最后一个位置设为pad_token_id
    labels[:-1] = input_ids[1:]
    labels[-1] = tokenizer.pad_token_id

    return input_ids, labels


def build_dataset(sample_length, tokenizer, window_size, corpus):
    """构建数据集"""
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(tokenizer, window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.stack(dataset_x), torch.stack(dataset_y)


def generate_sentence(openings, model, tokenizer, window_size, max_length=50):
    """生成文本"""
    model.eval()
    with torch.no_grad():
        generated = openings
        input_text = openings

        for _ in range(max_length):
            # 对当前文本进行编码
            encoding = tokenizer(
                input_text[-window_size:],  # 只取最后window_size个字符
                max_length=window_size,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            input_ids = encoding['input_ids']

            if torch.cuda.is_available():
                input_ids = input_ids.cuda()

            # 获取预测
            outputs = model(input_ids)
            next_token_logits = outputs[0, -1, :]  # 最后一个位置的预测

            # 采样策略
            if random.random() > 0.1:  # greedy
                next_token_id = torch.argmax(next_token_logits).item()
            else:  # sampling
                probs = torch.softmax(next_token_logits, dim=-1).cpu().numpy()
                next_token_id = np.random.choice(len(probs), p=probs)

            # 解码token
            next_token = tokenizer.decode([next_token_id])

            # 如果生成[SEP]或[PAD]，停止
            if next_token_id in [tokenizer.sep_token_id, tokenizer.pad_token_id]:
                break

            generated += next_token
            input_text += next_token

            # 如果生成了换行符，停止
            if '\n' in next_token:
                break

    return generated


def train(corpus_path, save_weight=True):
    """训练函数"""
    # 配置参数
    config = {
        "pretrain_model_path": r"C:\Users\Administrator\.cache\modelscope\hub\models\google-bert\bert-base-chinese",
        "max_length": 32,
        "window_size": 30,  # 实际文本长度，不包括特殊token
        "batch_size": 32,
        "train_sample": 50000,
        "epoch_num": 20,
        "learning_rate": 1e-4
    }

    # 加载语料
    corpus = load_corpus(corpus_path)
    print(f"语料长度: {len(corpus)}")

    # 初始化模型
    model = LanguageModel(config)
    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    print("开始训练...")
    for epoch in range(config["epoch_num"]):
        model.train()
        total_loss = 0
        batch_count = 0

        for batch in range(0, config["train_sample"], config["batch_size"]):
            # 构建batch
            x, y = build_dataset(
                min(config["batch_size"], config["train_sample"] - batch),
                model.tokenizer,
                config["window_size"],
                corpus
            )

            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()

            # 前向传播
            loss = model(x, labels=y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            if batch % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch}, Loss: {loss.item():.4f}")

        # 打印平均loss
        avg_loss = total_loss / batch_count
        print(f"=========\nEpoch {epoch + 1} 平均loss: {avg_loss:.4f}")

        # 生成示例文本
        model.eval()
        test_prompts = [
            "让他在半年之前，就不能做出",
            "李慕站在山路上，深深的呼吸",
            "这是一个美好的"
        ]

        for prompt in test_prompts:
            generated = generate_sentence(
                prompt, model, model.tokenizer,
                config["window_size"]
            )
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated}")
            print("-" * 50)

    # 保存模型
    if save_weight:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config
        }, 'bert_language_model.pth')
        print("模型已保存")


if __name__ == "__main__":
    # 训练模型
    train("corpus.txt", False)
