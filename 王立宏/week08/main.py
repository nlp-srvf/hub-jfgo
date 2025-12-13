# coding: utf-8

'''
main.py：使用训练数据集完成文本分类实验
'''

import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import time
import json
from datetime import datetime

from config import Config
from device_utils import set_device, optimize_for_device
from loader import get_data_loaders
from model import get_model
from evaluate import evaluate_model, evaluate_by_triplet_loss, print_evaluation_results

def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """
    训练一个epoch
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        optimizer: 优化器
        device: 计算设备
        epoch: 当前epoch
    Returns:
        avg_loss: 平均损失
    """
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    
    start_time = time.time()
    
    for batch_idx, batch in enumerate(train_loader):
        # 数据移到设备
        anchors = batch['anchors'].to(device)
        positives = batch['positives'].to(device)
        negatives = batch['negatives'].to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        
        # 前向传播
        anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives)
        
        # 计算损失
        loss = criterion(anchor_emb, positive_emb, negative_emb)
        
        # 反向传播
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # 更新参数
        optimizer.step()
        
        total_loss += loss.item()
        
        # 打印进度
        if batch_idx % 50 == 0:
            current_loss = loss.item()
            progress = batch_idx / num_batches * 100
            print(f'Epoch {epoch} [{batch_idx}/{num_batches} ({progress:.1f}%)] '
                  f'Loss: {current_loss:.6f}')
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - start_time
    
    print(f'Epoch {epoch} completed in {epoch_time:.2f}s, Average Loss: {avg_loss:.6f}')
    
    return avg_loss

def save_checkpoint(model, optimizer, epoch, loss, config):
    """
    保存模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        loss: 当前损失
        config: 配置字典
    """
    os.makedirs(config['model_path'], exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'config': config,
        'timestamp': datetime.now().isoformat()
    }
    
    checkpoint_path = os.path.join(config['model_path'], 'best_model.pth')
    torch.save(checkpoint, checkpoint_path)
    print(f'Model saved to {checkpoint_path}')

def load_checkpoint(model, optimizer, checkpoint_path):
    """
    加载模型检查点
    Args:
        model: 模型
        optimizer: 优化器
        checkpoint_path: 检查点路径
    Returns:
        epoch: 加载的epoch
        loss: 加载的损失
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

def main():
    """主训练函数"""
    print("开始训练文本匹配模型...")
    
    # 设置设备
    device = set_device()
    print(f"使用设备: {device}")
    
    # 设备优化
    optimize_for_device(device)
    
    # 获取数据加载器
    print("加载数据...")
    train_loader, valid_loader, vocab = get_data_loaders(Config)
    
    # 获取模型
    print("初始化模型...")
    model, criterion = get_model(len(vocab.word_to_idx), Config)
    model.to(device)
    
    # 设置优化器
    optimizer = optim.Adam(model.parameters(), lr=Config['learning_rate'])
    
    # 设置学习率调度器
    scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
    
    # 训练历史
    train_losses = []
    best_valid_metric = 0.0
    
    print(f"开始训练，总共 {Config['epoch']} 个epoch...")
    
    for epoch in range(1, Config['epoch'] + 1):
        print(f"\nEpoch {epoch}/{Config['epoch']}")
        print("-" * 50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        train_losses.append(train_loss)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"学习率: {current_lr:.6f}")
        
        # 验证
        print("\n验证模型...")
        valid_metrics, _, _, _ = evaluate_model(model, valid_loader, device)
        
        # 评估三元组损失
        triplet_loss, triplet_acc = evaluate_by_triplet_loss(model, train_loader, device)
        print(f"三元组损失: {triplet_loss:.6f}, 三元组准确率: {triplet_acc:.4f}")
        
        # 打印验证结果
        print_evaluation_results(valid_metrics, "Validation")
        
        # 保存最佳模型
        current_f1 = valid_metrics['f1']
        if current_f1 > best_valid_metric:
            best_valid_metric = current_f1
            save_checkpoint(model, optimizer, epoch, train_loss, Config)
            print(f"保存最佳模型 (F1: {current_f1:.4f})")
    
    print("\n训练完成!")
    print(f"最佳验证F1分数: {best_valid_metric:.4f}")
    
    # 保存训练历史
    history = {
        'train_losses': train_losses,
        'best_valid_f1': best_valid_metric,
        'config': Config,
        'vocab_size': len(vocab.word_to_idx)
    }
    
    history_path = os.path.join(Config['model_path'], 'training_history.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    print(f"训练历史保存到: {history_path}")


if __name__ == "__main__":
    main()

