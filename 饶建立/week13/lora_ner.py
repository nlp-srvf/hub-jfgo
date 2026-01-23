# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np
import os

# 导入你的配置和数据加载器
from config import Config
from data_loader import load_data


def evaluate(model, data_loader, device):
    """验证集评估函数"""
    model.eval()
    total_acc = 0
    total_count = 0

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # 模型前向传播
            outputs = model(batch_x)
            logits = outputs.logits  # shape: [batch, seq_len, num_classes]

            pred = torch.argmax(logits, dim=-1)  # shape: [batch, seq_len]

            # 计算准确率 (排除 padding 的 label -1)
            mask = batch_y != -1
            correct = (pred == batch_y) & mask

            total_acc += correct.sum().item()
            total_count += mask.sum().item()

    return total_acc / (total_count + 1e-9)


def main():
    # 1. 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. 加载数据
    print("Loading data...")
    train_loader = load_data(Config["train_data_path"], Config, shuffle=True)
    valid_loader = load_data(Config["valid_data_path"], Config, shuffle=False)

    # 3. 加载预训练模型 (Token Classification)
    print(f"Loading BERT model from {Config['bert_path']}...")
    model = AutoModelForTokenClassification.from_pretrained(
        Config["bert_path"],
        num_labels=Config["class_num"]
    )

    # 4. 配置 LoRA
    # task_type: 任务类型为 Token Classification
    # target_modules: BERT 的 Attention 层通常是 query, key, value
    # r: LoRA 的秩
    peft_config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["query", "key", "value"]  # 针对 BERT 结构的模块
    )

    # 5. 将模型封装为 PeftModel
    model = get_peft_model(model, peft_config)

    # 打印可训练参数信息，确认 LoRA 是否生效
    print("=" * 50)
    model.print_trainable_parameters()
    print("=" * 50)

    model = model.to(device)

    # 6. 定义优化器和损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config["learning_rate"])

    # 重要：你的 DataGenerator 对 label padding 填充的是 -1
    # CrossEntropyLoss 需要设置 ignore_index=-1 来忽略这些位置
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    # 7. 训练循环
    best_acc = 0
    save_path = os.path.join(Config["model_path"], "lora_best_model")

    for epoch in range(Config["epoch"]):
        model.train()
        total_loss = 0

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)  # shape: [batch, seq_len]

            optimizer.zero_grad()

            # 前向传播
            outputs = model(batch_x)
            logits = outputs.logits  # [batch, seq_len, class_num]

            # 计算 Loss
            # 需要将 logits 展平为 [N, C], labels 展平为 [N]
            loss = criterion(logits.view(-1, Config["class_num"]), batch_y.view(-1))

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch {epoch + 1}/{Config['epoch']} | Step {step} | Loss: {loss.item():.4f}")

        # 验证
        avg_loss = total_loss / len(train_loader)
        acc = evaluate(model, valid_loader, device)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_loss:.4f}, Valid Acc: {acc:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            print(f"New best accuracy! Saving model to {save_path}")
            model.save_pretrained(save_path)

    print("Training finished.")


if __name__ == "__main__":
    # 创建输出目录
    if not os.path.exists(Config["model_path"]):
        os.makedirs(Config["model_path"])

    main()