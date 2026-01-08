说明：用两个BERT堆叠的方式完成Seq2Seq 训练，效果不好，掌握原理



# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import json
import random
import os

from torch import device
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader

"""
基于BERT的SFT Seq2Seq模型
"""


class SFTDataset(Dataset):
    """SFT数据集类"""

    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # 构建编码器输入
        if item["input"]:
            encoder_text = f"instruction: {item['instruction']} input: {item['input']}"
        else:
            encoder_text = f"instruction: {item['instruction']}"

        # 解码器输入和目标
        decoder_text = item["output"]

        # 编码器输入编码
        encoder_input = self.tokenizer(
            encoder_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 解码器输入编码（带[CLS]）
        decoder_input = self.tokenizer(
            decoder_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 解码器目标编码（不带[CLS]，移位一位）
        target_input = self.tokenizer(
            decoder_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 创建注意力掩码（解码器只能看到之前的信息）
        decoder_mask = self.create_decoder_mask(self.max_length)

        # 创建标签（将输入向右移动一位）
        labels = target_input['input_ids'].squeeze(0).clone()

        return {
            'encoder_input_ids': encoder_input['input_ids'].squeeze(0),
            'encoder_attention_mask': encoder_input['attention_mask'].squeeze(0),
            'decoder_input_ids': decoder_input['input_ids'].squeeze(0),
            'decoder_attention_mask': decoder_mask,
            'labels': labels
        }

    def create_decoder_mask(self, size):
        """创建下三角注意力掩码"""
        mask = torch.tril(torch.ones(size, size))
        return mask


class SFTSeq2SeqModel(nn.Module):
    """SFT Seq2Seq模型"""

    def __init__(self, pretrain_model_path, vocab,tokenizer):
        super(SFTSeq2SeqModel, self).__init__()

        # 共享BERT编码器（也可使用两个独立的BERT）
        self.encoder = BertModel.from_pretrained(
            pretrain_model_path,
            return_dict=False,
            attn_implementation='eager'
        )
        self.decoder = BertModel.from_pretrained(
            pretrain_model_path,
            return_dict=False,
            attn_implementation='eager'
        )

        # 分类头
        self.classify = nn.Linear(768, vocab)

        # 编码器到解码器的连接层（可选）
        self.encoder2decoder = nn.Linear(768, 768)

    def forward(self, encoder_input_ids, encoder_attention_mask,
                decoder_input_ids, decoder_attention_mask, labels=None):

        # 编码器部分
        encoder_outputs, _ = self.encoder(
            encoder_input_ids,
            attention_mask=encoder_attention_mask
        )

        # 将编码器输出转换到解码器空间
        encoder_outputs = self.encoder2decoder(encoder_outputs)

        # 解码器部分
        decoder_outputs, _ = self.decoder(
            decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs,
            encoder_attention_mask=encoder_attention_mask
        )

        # 分类层
        logits = self.classify(decoder_outputs)

        if labels is not None:
            # 计算损失（忽略padding token）
            loss_fn = nn.CrossEntropyLoss(ignore_index=0)
            # 将logits和labels重塑为2D
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            return loss, logits
        else:
            return logits

    def generate(self, encoder_input_ids, encoder_attention_mask,
                 max_length=50, temperature=1.0):
        """生成文本"""
        self.eval()

        # 编码输入
        encoder_outputs, _ = self.encoder(
            encoder_input_ids,
            attention_mask=encoder_attention_mask
        )
        encoder_outputs = self.encoder2decoder(encoder_outputs)

        # 初始化解码器输入（[CLS] token）
        batch_size = encoder_input_ids.size(0)
        decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(encoder_input_ids.device) * 101  # [CLS]
        # decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long).to(device) * tokenizer.cls_token_id
        # 自回归生成
        for _ in range(max_length):
            # 创建解码器掩码
            seq_len = decoder_input_ids.size(1)
            decoder_attention_mask = torch.ones((batch_size, seq_len), device=encoder_input_ids.device)
            # decoder_attention_mask = torch.tril(
            #     torch.ones((batch_size, seq_len), device=encoder_input_ids.device)
            # )
            # 解码
            decoder_outputs, _ = self.decoder(
                decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=encoder_outputs,
                encoder_attention_mask=encoder_attention_mask
            )

            # 获取下一个token的logits
            next_token_logits = self.classify(decoder_outputs[:, -1, :]) / temperature

            # 采样下一个token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 添加到输入序列
            decoder_input_ids = torch.cat([decoder_input_ids, next_token], dim=-1)

            # 如果生成了[SEP] token则停止
            if (next_token == 102).all():  # [SEP] token
                break

        return decoder_input_ids


def prepare_data():
    """准备SFT数据"""
    data = [
        {
            "instruction": "请介绍一下人工智能。",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用具有多个层（深度）的神经网络来学习数据的复杂模式和表示。它模仿人脑的神经网络结构。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "input": "",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "input": "",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "input": "",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        }
    ]
    return data
def build_vocab(vocab_path):
    vocab = {"<pad>":0}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            char = line[:-1]       #去掉结尾换行符
            vocab[char] = index + 1 #留出0位给pad token
    return vocab

def train():
    """训练函数"""
    # 调整参数
    epoch_num = 30  # 增加训练轮数
    batch_size = 4
    learning_rate = 3e-5  # 降低学习率
    max_length = 64  # 增加序列长度

    # 加载预训练模型和tokenizer
    pretrain_model_path = r'C:\Users\Administrator\.cache\modelscope\hub\models\google-bert\bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(pretrain_model_path)

    # 准备数据
    data = prepare_data()
    dataset = SFTDataset(data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # vocab = build_vocab("vocab.txt")
    # 初始化模型
    model = SFTSeq2SeqModel(pretrain_model_path, tokenizer.vocab_size)

    if torch.cuda.is_available():
        model = model.cuda()

    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    print("开始训练...")
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0

        for batch_idx, batch in enumerate(dataloader):
            # 移动到GPU
            if torch.cuda.is_available():
                for key in batch:
                    batch[key] = batch[key].cuda()

            # 前向传播
            loss, _ = model(
                encoder_input_ids=batch['encoder_input_ids'],
                encoder_attention_mask=batch['encoder_attention_mask'],
                decoder_input_ids=batch['decoder_input_ids'],
                decoder_attention_mask=batch['decoder_attention_mask'],
                labels=batch['labels']
            )

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 2 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}")

        # 测试生成
        test_generation(model, tokenizer)

    # 保存模型
    torch.save(model.state_dict(), "sft_seq2seq_model.pth")
    print("模型已保存")


def test_generation(model, tokenizer):
    """测试生成效果"""
    model.eval()

    test_cases = [
        {"instruction": "请介绍一下人工智能。", "input": ""},
        {"instruction": "什么是机器学习？", "input": ""},
    ]

    with torch.no_grad():
        for test_case in test_cases:
            # 准备编码器输入
            if test_case["input"]:
                encoder_text = f"instruction: {test_case['instruction']} input: {test_case['input']}"
            else:
                encoder_text = f"instruction: {test_case['instruction']}"

            encoder_input = tokenizer(
                encoder_text,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            if torch.cuda.is_available():
                encoder_input = {k: v.cuda() for k, v in encoder_input.items()}

            # 生成文本
            generated_ids = model.generate(
                encoder_input_ids=encoder_input['input_ids'],
                encoder_attention_mask=encoder_input['attention_mask'],
                max_length=50
            )

            # 解码生成结果
            generated_text = tokenizer.decode(
                generated_ids[0].cpu().numpy(),
                skip_special_tokens=True
            )

            print(f"输入: {test_case['instruction']}")
            print(f"生成: {generated_text}")
            print("-" * 50)


if __name__ == "__main__":
    train()

