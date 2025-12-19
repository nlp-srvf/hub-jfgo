# coding: utf-8

'''
main.py：使用训练数据集完成NER实验
'''

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import time
import json
from typing import Dict, Any, Optional
import argparse

from config import Config
from device_utils import set_device, print_device_info, optimize_for_device
from loader import NERDataLoader
from model import create_model, BertCRFForNER
from evaluate import evaluate_model, NEREvaluator


class NERTrainer:
    """
    NER模型训练器
    负责模型训练、验证和保存
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 配置字典
        """
        self.config = config
        self.device = set_device()
        self.label2id, self.id2label = self._load_label_mapping()
        
        # 创建保存目录
        os.makedirs(config['model_path'], exist_ok=True)
        
        # 初始化组件
        self.data_loader = NERDataLoader(config)
        self.model = create_model(config)
        self.model.to(self.device)
        
        # 设置优化器
        self.optimizer = self._setup_optimizer()
        
        # 设置学习率调度器
        self.scheduler = self._setup_scheduler()
        
        # 训练状态
        self.global_step = 0
        self.best_f1 = 0.0
        self.train_losses = []
        self.val_metrics = []
    
    def _load_label_mapping(self) -> tuple:
        """加载标签映射"""
        with open(self.config['schema_path'], 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        label2id = schema
        id2label = {v: k for k, v in schema.items()}
        
        return label2id, id2label
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """
        设置优化器
        
        Returns:
            优化器实例
        """
        if self.config['optimizer'].lower() == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=1e-5
            )
        elif self.config['optimizer'].lower() == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=1e-5
            )
        else:
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                momentum=0.9
            )
        
        return optimizer
    
    def _setup_scheduler(self) -> Optional[object]:
        """
        设置学习率调度器
        
        Returns:
            学习率调度器实例
        """
        from transformers import get_linear_schedule_with_warmup
        
        # 获取训练数据加载器计算总步数
        train_dataloader, _ = self.data_loader.get_dataloaders()
        total_steps = len(train_dataloader) * self.config['epoch']
        warmup_steps = int(0.1 * total_steps)  # 10%的热身
        
        scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        return scheduler
    
    def train_epoch(self, train_dataloader: DataLoader, epoch: int) -> float:
        """
        训练一个epoch
        
        Args:
            train_dataloader: 训练数据加载器
            epoch: 当前epoch
            
        Returns:
            平均损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_dataloader)
        
        print(f"🚀 开始第 {epoch + 1}/{self.config['epoch']} 轮训练...")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            token_type_ids = batch['token_type_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # 清零梯度
            self.optimizer.zero_grad()
            
            # 前向传播
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs['loss']
            
            # CRF返回的是每个样本的loss，需要取平均
            if loss.dim() > 0:
                loss = loss.mean()
            
            # 检查数值稳定性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ 检测到不稳定的损失值: {loss.item()}，跳过此批次")
                continue
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪（防止梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            self.scheduler.step()
            
            # 记录损失
            total_loss += loss.item()
            self.global_step += 1
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                avg_loss = total_loss / (batch_idx + 1)
                current_lr = self.scheduler.get_last_lr()[0]
                print(f"📊 Epoch {epoch + 1}, Batch {batch_idx + 1}/{num_batches}, "
                      f"Loss: {loss.item():.4f}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.2e}")
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        
        return avg_loss
    
    def validate(self, val_dataloader: DataLoader) -> Dict[str, float]:
        """
        验证模型
        
        Args:
            val_dataloader: 验证数据加载器
            
        Returns:
            验证指标
        """
        print("🔍 开始验证...")
        
        metrics = evaluate_model(self.model, val_dataloader, self.device, self.id2label)
        self.val_metrics.append(metrics)
        
        return metrics
    
    def save_model(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """
        保存模型
        
        Args:
            epoch: 当前epoch
            metrics: 验证指标
            is_best: 是否是最佳模型
        """
        # 保存模型检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'metrics': metrics,
            'config': self.config
        }
        
        # 保存最新模型
        latest_path = os.path.join(self.config['model_path'], 'latest_model.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.config['model_path'], 'best_model.pt')
            torch.save(checkpoint, best_path)
            print(f"🏆 保存最佳模型到: {best_path}")
        
        # 保存训练历史
        history = {
            'train_losses': self.train_losses,
            'val_metrics': self.val_metrics
        }
        history_path = os.path.join(self.config['model_path'], 'training_history.json')
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        
        print(f"💾 模型已保存到: {latest_path}")
    
    def train(self):
        """
        完整的训练流程
        """
        print("🎯 开始训练BERT+CRF NER模型...")
        print(f"📋 配置信息: {json.dumps(self.config, ensure_ascii=False, indent=2)}")
        
        # 获取数据加载器
        train_dataloader, val_dataloader = self.data_loader.get_dataloaders()
        
        print(f"📊 训练批次数: {len(train_dataloader)}")
        print(f"📊 验证批次数: {len(val_dataloader)}")
        print(f"🏷️  标签数量: {len(self.label2id)}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.config['epoch']):
                # 训练一个epoch
                train_loss = self.train_epoch(train_dataloader, epoch)
                
                # 验证
                val_metrics = self.validate(val_dataloader)
                current_f1 = val_metrics['overall_f1']
                
                # 检查是否是最佳模型
                is_best = current_f1 > self.best_f1
                if is_best:
                    self.best_f1 = current_f1
                
                # 保存模型
                self.save_model(epoch, val_metrics, is_best)
                
                # 打印epoch总结
                print(f"\n📈 Epoch {epoch + 1} 总结:")
                print(f"📉 训练损失: {train_loss:.4f}")
                print(f"🎯 验证F1: {current_f1:.4f}")
                print(f"🏆 最佳F1: {self.best_f1:.4f}")
                print("-" * 60)
        
        except KeyboardInterrupt:
            print("\n⚠️  训练被用户中断")
        
        except Exception as e:
            print(f"\n❌ 训练过程中出现错误: {e}")
            raise
        
        finally:
            # 计算训练时间
            training_time = time.time() - start_time
            print(f"\n⏱️  训练完成，耗时: {training_time/60:.2f} 分钟")
            print(f"🏆 最佳验证F1值: {self.best_f1:.4f}")
    
    def load_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.global_step = checkpoint['global_step']
            print(f"✅ 模型已从 {model_path} 加载")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='BERT+CRF NER模型训练')
    
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径（可选）')
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], default='train',
                       help='运行模式：train（训练）或 eval（评估）')
    parser.add_argument('--model_path', type=str, default=None,
                       help='模型加载路径（评估模式使用）')
    parser.add_argument('--epoch', type=int, default=None,
                       help='训练轮数（覆盖配置文件）')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小（覆盖配置文件）')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率（覆盖配置文件）')
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 打印设备信息
    print_device_info()
    
    # 优化设备设置
    device = set_device()
    optimize_for_device(device)
    
    # 更新配置
    config = Config.copy()
    if args.epoch:
        config['epoch'] = args.epoch
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        # 训练模式
        trainer = NERTrainer(config)
        trainer.train()
    
    elif args.mode == 'eval':
        # 评估模式
        if args.model_path is None:
            # 使用最佳模型
            model_path = os.path.join(config['model_path'], 'best_model.pt')
        else:
            model_path = args.model_path
        
        if not os.path.exists(model_path):
            print(f"❌ 模型文件不存在: {model_path}")
            return
        
        # 创建训练器并加载模型
        trainer = NERTrainer(config)
        trainer.load_model(model_path)
        
        # 获取验证数据并评估
        _, val_dataloader = trainer.data_loader.get_dataloaders()
        metrics = evaluate_model(trainer.model, val_dataloader, device, trainer.id2label)
        
        # 保存评估结果
        eval_result_path = os.path.join(config['model_path'], 'evaluation_results.json')
        with open(eval_result_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        print(f"📄 评估结果已保存到: {eval_result_path}")


if __name__ == "__main__":
    main()
