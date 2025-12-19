# -*- coding: utf-8 -*-

import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import BertNERModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序 - BERT版本
"""


def set_seed(seed=42):
    """设置随机种子以确保可复现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main(config):
    # 设置随机种子
    set_seed(42)
    
    # 创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    
    # 加载训练数据
    logger.info("Loading training data...")
    train_data = load_data(config["train_data_path"], config)
    logger.info(f"Training data loaded. Total batches: {len(train_data)}")
    
    # 加载模型
    logger.info("Loading BERT model...")
    model = BertNERModel(config)
    logger.info("BERT model loaded.")
    
    # 标识是否使用GPU
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("GPU可以使用，迁移模型至GPU")
        model = model.cuda()
    
    # 加载优化器
    optimizer = choose_optimizer(config, model)
    
    # 加载效果测试类
    evaluator = Evaluator(config, model, logger)
    
    # 训练
    best_f1 = 0
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("=" * 50)
        logger.info("Epoch %d begin" % epoch)
        train_loss = []
        
        for index, batch_data in enumerate(train_data):
            optimizer.zero_grad()
            
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            
            # BERT模型需要input_ids, attention_mask, labels
            input_ids, attention_mask, labels = batch_data
            
            # 前向传播计算损失
            loss = model(input_ids, attention_mask, labels)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss.append(loss.item())
            
            # 打印训练进度
            if index % max(1, int(len(train_data) / 2)) == 0:
                logger.info("Batch %d/%d, Loss: %f" % (index, len(train_data), loss.item()))
        
        logger.info("Epoch %d average loss: %f" % (epoch, np.mean(train_loss)))
        
        # 评估
        f1 = evaluator.eval(epoch)
        
        # 保存最佳模型
        if f1 > best_f1:
            best_f1 = f1
            model_path = os.path.join(config["model_path"], "best_model.pth")
            torch.save(model.state_dict(), model_path)
            logger.info(f"Best model saved with F1: {best_f1:.4f}")
    
    # 保存最终模型
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    logger.info(f"Final model saved to {model_path}")
    
    return model, train_data


if __name__ == "__main__":
    model, train_data = main(Config)

