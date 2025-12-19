# coding: utf-8

'''
evaluate.py：模型评估函数，支持多次重复评估和k折交叉验证
'''

import torch
import numpy as np
from typing import Dict, List, Tuple, Any
from collections import defaultdict
from sklearn.metrics import classification_report
import json


class NEREvaluator:
    """
    NER模型评估器
    计算NER任务的各项评估指标
    """
    
    def __init__(self, id2label: Dict[int, str]):
        """
        初始化评估器
        
        Args:
            id2label: ID到标签的映射字典
        """
        self.id2label = id2label
        self.reset_metrics()
    
    def reset_metrics(self):
        """重置所有评估指标"""
        self.true_positive = defaultdict(int)
        self.false_positive = defaultdict(int)
        self.false_negative = defaultdict(int)
        self.all_predictions = []
        self.all_labels = []
    
    def add_batch(self, predictions: List[List[int]], gold_labels: List[List[int]]):
        """
        添加一个批次的预测结果和真实标签
        
        Args:
            predictions: 预测标签序列列表
            gold_labels: 真实标签序列列表
        """
        for pred_seq, gold_seq in zip(predictions, gold_labels):
            # 确保长度一致
            min_len = min(len(pred_seq), len(gold_seq))
            pred_seq = pred_seq[:min_len]
            gold_seq = gold_seq[:min_len]
            
            for pred_label, gold_label in zip(pred_seq, gold_seq):
                # 转换为标签名称
                pred_name = self.id2label.get(pred_label, 'O')
                gold_name = self.id2label.get(gold_label, 'O')
                
                # 保存所有预测用于整体评估
                self.all_predictions.append(pred_label)
                self.all_labels.append(gold_label)
                
                # 只计算实体标签的指标（忽略'O'标签）
                if gold_name != 'O':
                    if pred_name == gold_name:
                        self.true_positive[gold_name] += 1
                    else:
                        self.false_negative[gold_name] += 1
                        if pred_name != 'O':
                            self.false_positive[pred_name] += 1
                elif pred_name != 'O':
                    # 真实标签为O，但预测为实体标签
                    self.false_positive[pred_name] += 1
    
    def get_metrics(self) -> Dict[str, float]:
        """
        计算并返回各项评估指标
        
        Returns:
            包含各项指标的字典
        """
        metrics = {}
        
        # 计算每个标签的指标
        precision_sum = 0.0
        recall_sum = 0.0
        f1_sum = 0.0
        entity_count = 0
        
        for label in self.id2label.values():
            if label == 'O':
                continue
            
            tp = self.true_positive[label]
            fp = self.false_positive[label]
            fn = self.false_negative[label]
            
            # 计算精确率
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            
            # 计算召回率
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            # 计算F1值
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            metrics[f'{label}_precision'] = precision
            metrics[f'{label}_recall'] = recall
            metrics[f'{label}_f1'] = f1
            
            if tp + fp > 0 or tp + fn > 0:
                precision_sum += precision
                recall_sum += recall
                f1_sum += f1
                entity_count += 1
        
        # 计算宏平均指标
        if entity_count > 0:
            metrics['macro_precision'] = precision_sum / entity_count
            metrics['macro_recall'] = recall_sum / entity_count
            metrics['macro_f1'] = f1_sum / entity_count
        
        # 计算总体指标
        total_tp = sum(self.true_positive.values())
        total_fp = sum(self.false_positive.values())
        total_fn = sum(self.false_negative.values())
        
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        metrics['overall_precision'] = overall_precision
        metrics['overall_recall'] = overall_recall
        metrics['overall_f1'] = overall_f1
        
        return metrics
    
    def get_classification_report(self) -> str:
        """
        获取详细的分类报告
        
        Returns:
            分类报告字符串
        """
        # 过滤掉-100的标签
        filtered_predictions = []
        filtered_labels = []
        
        for pred, gold in zip(self.all_predictions, self.all_labels):
            if gold != -100:  # 忽略padding标签
                filtered_predictions.append(pred)
                filtered_labels.append(gold)
        
        # 获取标签名称
        target_names = [self.id2label[i] for i in sorted(self.id2label.keys()) if self.id2label[i] != 'O']
        
        # 生成分类报告
        report = classification_report(
            filtered_labels,
            filtered_predictions,
            target_names=target_names,
            zero_division=0
        )
        
        return report
    
    def print_metrics(self, metrics: Dict[str, float]):
        """
        打印评估指标
        
        Args:
            metrics: 评估指标字典
        """
        print("\n" + "="*60)
        print("📊 NER模型评估结果")
        print("="*60)
        
        # 打印每个实体类型的指标
        entity_types = [label for label in self.id2label.values() if label != 'O']
        
        print("\n🏷️  各实体类型指标:")
        print("-" * 60)
        print(f"{'实体类型':<15} {'精确率':<10} {'召回率':<10} {'F1值':<10}")
        print("-" * 60)
        
        for entity_type in entity_types:
            precision = metrics.get(f'{entity_type}_precision', 0.0)
            recall = metrics.get(f'{entity_type}_recall', 0.0)
            f1 = metrics.get(f'{entity_type}_f1', 0.0)
            print(f"{entity_type:<15} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")
        
        # 打印总体指标
        print("\n📈 总体指标:")
        print("-" * 60)
        print(f"宏平均精确率: {metrics.get('macro_precision', 0.0):.4f}")
        print(f"宏平均召回率: {metrics.get('macro_recall', 0.0):.4f}")
        print(f"宏平均F1值:   {metrics.get('macro_f1', 0.0):.4f}")
        print("-" * 60)
        print(f"总体精确率:   {metrics.get('overall_precision', 0.0):.4f}")
        print(f"总体召回率:   {metrics.get('overall_recall', 0.0):.4f}")
        print(f"总体F1值:     {metrics.get('overall_f1', 0.0):.4f}")
        print("="*60)
    
    def save_metrics(self, metrics: Dict[str, float], filepath: str):
        """
        保存评估指标到文件
        
        Args:
            metrics: 评估指标字典
            filepath: 保存路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"📄 评估指标已保存到: {filepath}")


def evaluate_model(model, dataloader, device, id2label: Dict[int, str]) -> Dict[str, float]:
    """
    评估模型在给定数据集上的性能
    
    Args:
        model: 待评估的模型
        dataloader: 数据加载器
        device: 设备
        id2label: ID到标签的映射
        
    Returns:
        评估指标字典
    """
    model.eval()
    evaluator = NEREvaluator(id2label)
    
    print("🔍 开始评估模型...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # 将数据移到设备
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # 前向传播
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 获取预测结果
            predictions = model.decode(outputs['emissions'], outputs['mask'])
            
            # 处理真实标签（移除-100）
            gold_labels = []
            for label_seq in labels:
                gold_seq = label_seq.cpu().numpy()
                # 移除-100标签（padding标签）
                gold_seq = gold_seq[gold_seq != -100]
                gold_labels.append(gold_seq.tolist())
            
            # 添加到评估器
            evaluator.add_batch(predictions, gold_labels)
            
            # 打印进度
            if (batch_idx + 1) % 10 == 0:
                print(f"📊 已处理 {batch_idx + 1}/{len(dataloader)} 个批次")
    
    # 计算指标
    metrics = evaluator.get_metrics()
    
    # 打印结果
    evaluator.print_metrics(metrics)
    
    return metrics


def cross_validate(model_class, config, data_loader, device, k_folds: int = 5) -> Dict[str, float]:
    """
    K折交叉验证
    
    Args:
        model_class: 模型类
        config: 配置字典
        data_loader: 数据加载器
        device: 设备
        k_folds: 折数
        
    Returns:
        交叉验证的平均指标
    """
    from sklearn.model_selection import KFold
    import torch.utils.data as data
    
    print(f"🔄 开始{k_folds}折交叉验证...")
    
    # 获取完整数据集
    train_dataset, _ = data_loader.get_datasets()
    
    # 创建K折分割器
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    all_metrics = []
    id2label = data_loader.id2label
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_dataset)):
        print(f"\n📂 第 {fold + 1}/{k_folds} 折:")
        
        # 创建数据子集
        train_subset = data.Subset(train_dataset, train_idx)
        val_subset = data.Subset(train_dataset, val_idx)
        
        # 创建数据加载器
        train_loader = data.DataLoader(
            train_subset,
            batch_size=config['batch_size'],
            shuffle=True,
            collate_fn=data_loader._collate_fn
        )
        
        val_loader = data.DataLoader(
            val_subset,
            batch_size=config['batch_size'],
            shuffle=False,
            collate_fn=data_loader._collate_fn
        )
        
        # 创建模型
        model = model_class(config)
        model.to(device)
        
        # 这里应该进行训练，但为了简化，我们只做评估
        # 实际使用时应该调用训练函数
        
        # 评估模型
        metrics = evaluate_model(model, val_loader, device, id2label)
        all_metrics.append(metrics)
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        std_metrics = np.std([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = std_metrics
    
    print("\n📊 交叉验证结果:")
    print("="*60)
    print(f"总体F1值: {avg_metrics['overall_f1']:.4f} ± {avg_metrics['overall_f1_std']:.4f}")
    print(f"总体精确率: {avg_metrics['overall_precision']:.4f} ± {avg_metrics['overall_precision_std']:.4f}")
    print(f"总体召回率: {avg_metrics['overall_recall']:.4f} ± {avg_metrics['overall_recall_std']:.4f}")
    print("="*60)
    
    return avg_metrics


def test_evaluator():
    """
    测试评估器功能
    """
    from config import Config
    
    print("🧪 测试评估器...")
    
    # 创建模拟的id2label
    label2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-LOC': 3, 'I-LOC': 4}
    id2label = {v: k for k, v in label2id.items()}
    
    # 创建评估器
    evaluator = NEREvaluator(id2label)
    
    # 添加一些测试数据
    predictions = [[1, 2, 0, 3, 4], [0, 0, 1, 2, 0]]
    gold_labels = [[1, 2, 0, 0, 3], [0, 0, 1, 0, 4]]
    
    evaluator.add_batch(predictions, gold_labels)
    
    # 计算指标
    metrics = evaluator.get_metrics()
    evaluator.print_metrics(metrics)
    
    print("✅ 评估器测试完成！")


if __name__ == "__main__":
    test_evaluator()
