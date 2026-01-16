
# train.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from config import Config
from dataset import get_dataloader
from model import get_model

def train():
    # 1. 初始化配置
    cfg = Config()

    # 2. 准备数据
    dataloader, tokenizer = get_dataloader(cfg)

    # 3. 准备模型
    model = get_model(cfg)
    model.train()

    # 4. 优化器
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate)

    print("Start Training...")

    for epoch in range(cfg.epoch):
        total_loss = 0
        for step, batch in enumerate(dataloader):
            # 将数据搬运到 GPU/CPU
            input_ids = batch['input_ids'].to(cfg.device)
            attention_mask = batch['attention_mask'].to(cfg.device)
            labels = batch['labels'].to(cfg.device)

            # 清空梯度
            optimizer.zero_grad()

            # 前向传播
            # BertLMHeadModel 内部会自动处理 Labels 的移位计算 Loss
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0:
                print(f"Epoch [{epoch +1}/{cfg.epoch}], Step [{step}/{len(dataloader)}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch +1} finished. Average Loss: {avg_loss:.4f}")

    # 5. 保存模型
    torch.save(model.state_dict(), cfg.model_save_path)
    print(f"Model saved to {cfg.model_save_path}")

if __name__ == "__main__":
    train()