# coding: utf-8

'''
模型效果测试
'''

import os
import torch
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict

from config import Config
from device_utils import set_device
from loader import Vocabulary, TextMatchingEvalDataset
from model import get_model

class TextMatchingPredictor:
    """文本匹配预测器"""
    def __init__(self, model_path, config):
        """
        初始化预测器
        Args:
            model_path: 模型文件路径
            config: 配置字典
        """
        self.config = config
        self.device = set_device()
        
        # 加载词汇表
        self.vocab = Vocabulary(config['vocab_path'])
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 构建类别原型
        self.class_prototypes = {}
        self._build_class_prototypes()
    
    def _load_model(self, model_path):
        """
        加载训练好的模型
        Args:
            model_path: 模型文件路径
        Returns:
            model: 加载的模型
        """
        model, _ = get_model(len(self.vocab.word_to_idx), self.config)
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"成功加载模型: {model_path}")
            print(f"训练epoch: {checkpoint['epoch']}")
            print(f"训练损失: {checkpoint['loss']:.6f}")
        else:
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        return model
    
    def _build_class_prototypes(self):
        """
        构建每个类别的原型向量
        """
        print("构建类别原型...")
        
        # 加载训练数据构建原型
        with open(self.config['train_data_path'], 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        with open(self.config['schema_path'], 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        idx_to_label = {v: k for k, v in schema.items()}
        
        # 收集每个类别的所有文本
        class_texts = defaultdict(list)
        
        for line in lines:
            item = json.loads(line.strip())
            questions = item['questions']
            target = item['target']
            label = schema[target]
            
            for question in questions:
                class_texts[label].append(question)
        
        # 计算每个类别的平均嵌入
        with torch.no_grad():
            for label, texts in class_texts.items():
                embeddings = []
                
                for text in texts:
                    indices = self.vocab.text_to_indices(text, self.config['max_length'])
                    input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
                    
                    embedding = self.model.encode(input_tensor)
                    embeddings.append(embedding.cpu().numpy())
                
                # 计算平均嵌入作为类别原型
                if embeddings:
                    self.class_prototypes[label] = np.mean(embeddings, axis=0)
        
        print(f"构建了 {len(self.class_prototypes)} 个类别原型")
    
    def predict(self, text, return_top_k=3):
        """
        预测单个文本的类别
        Args:
            text: 输入文本
            return_top_k: 返回top-k个预测结果
        Returns:
            predictions: 预测结果列表
        """
        # 文本预处理
        indices = self.vocab.text_to_indices(text, self.config['max_length'])
        input_tensor = torch.tensor([indices], dtype=torch.long).to(self.device)
        
        # 获取文本嵌入
        with torch.no_grad():
            text_embedding = self.model.encode(input_tensor).cpu().numpy()
        
        # 计算与所有类别原型的相似度
        similarities = {}
        for label, prototype in self.class_prototypes.items():
            sim = cosine_similarity(text_embedding.reshape(1, -1), 
                                 prototype.reshape(1, -1))[0][0]
            similarities[label] = sim
        
        # 排序并返回top-k
        sorted_similarities = sorted(similarities.items(), 
                                  key=lambda x: x[1], reverse=True)
        
        predictions = []
        for i, (label, similarity) in enumerate(sorted_similarities[:return_top_k]):
            predictions.append({
                'label': label,
                'target': self._label_to_target(label),
                'similarity': float(similarity),
                'rank': i + 1
            })
        
        return predictions
    
    def _label_to_target(self, label):
        """
        将标签索引转换为目标名称
        Args:
            label: 标签索引
        Returns:
            target: 目标名称
        """
        with open(self.config['schema_path'], 'r', encoding='utf-8') as f:
            schema = json.load(f)
        
        idx_to_label = {v: k for k, v in schema.items()}
        return idx_to_label.get(label, "unknown")
    
    def batch_predict(self, texts, return_top_k=1):
        """
        批量预测
        Args:
            texts: 文本列表
            return_top_k: 返回top-k个预测结果
        Returns:
            batch_predictions: 批量预测结果
        """
        batch_predictions = []
        
        for text in texts:
            predictions = self.predict(text, return_top_k)
            batch_predictions.append(predictions)
        
        return batch_predictions
    
    def evaluate_on_dataset(self, dataset_path):
        """
        在数据集上评估模型
        Args:
            dataset_path: 数据集路径
        Returns:
            metrics: 评估指标
        """
        print(f"在数据集 {dataset_path} 上评估模型...")
        
        # 加载数据
        dataset = TextMatchingEvalDataset(
            dataset_path,
            self.config['schema_path'],
            self.vocab,
            self.config['max_length']
        )
        
        correct_predictions = 0
        total_samples = len(dataset)
        top_1_correct = 0
        top_3_correct = 0
        
        for i, item in enumerate(dataset):
            if i % 100 == 0:
                print(f"处理进度: {i}/{total_samples} ({i/total_samples*100:.1f}%)")
            
            text = item['text']
            true_label = item['label'].item()
            true_target = item['target']
            
            # 预测
            predictions = self.predict(text, return_top_k=3)
            
            # Top-1准确率
            pred_label = predictions[0]['label']
            if pred_label == true_label:
                top_1_correct += 1
            
            # Top-3准确率
            pred_labels = [p['label'] for p in predictions]
            if true_label in pred_labels:
                top_3_correct += 1
        
        top_1_accuracy = top_1_correct / total_samples
        top_3_accuracy = top_3_correct / total_samples
        
        metrics = {
            'total_samples': total_samples,
            'top_1_correct': top_1_correct,
            'top_3_correct': top_3_correct,
            'top_1_accuracy': top_1_accuracy,
            'top_3_accuracy': top_3_accuracy
        }
        
        print(f"Top-1准确率: {top_1_accuracy:.4f}")
        print(f"Top-3准确率: {top_3_accuracy:.4f}")
        
        return metrics

def interactive_test(predictor):
    """
    交互式测试
    Args:
        predictor: 预测器实例
    """
    print("\n" + "="*50)
    print("文本匹配模型交互式测试")
    print("="*50)
    print("输入文本进行预测，输入 'quit' 退出")
    print("="*50)
    
    while True:
        text = input("\n请输入文本: ").strip()
        
        if text.lower() == 'quit':
            break
        
        if not text:
            continue
        
        try:
            # 预测
            predictions = predictor.predict(text, return_top_k=3)
            
            print(f"\n输入: {text}")
            print("\n预测结果:")
            print("-" * 40)
            
            for pred in predictions:
                print(f"第{pred['rank']}名: {pred['target']}")
                print(f"相似度: {pred['similarity']:.4f}")
                print("-" * 40)
                
        except Exception as e:
            print(f"预测出错: {e}")

def main():
    """主函数"""
    model_path = os.path.join(Config['model_path'], 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        print("请先训练模型!")
        return
    
    # 创建预测器
    predictor = TextMatchingPredictor(model_path, Config)
    
    # 选择模式
    print("\n请选择测试模式:")
    print("1. 交互式测试")
    print("2. 验证集评估")
    print("3. 自定义文本测试")
    
    choice = input("请输入选择 (1-3): ").strip()
    
    if choice == "1":
        interactive_test(predictor)
    elif choice == "2":
        # 验证集评估
        metrics = predictor.evaluate_on_dataset(Config['valid_data_path'])
        
        # 保存评估结果
        results_path = os.path.join(Config['model_path'], 'test_results.json')
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"评估结果保存到: {results_path}")
        
    elif choice == "3":
        # 自定义文本测试
        test_texts = [
            "查一下我的话费",
            "怎么修改密码", 
            "开通来电显示",
            "查询积分",
            "设置呼叫转移"
        ]
        
        print("\n测试文本:")
        print("-" * 50)
        
        for text in test_texts:
            predictions = predictor.predict(text, return_top_k=3)
            print(f"\n输入: {text}")
            print(f"预测: {predictions[0]['target']} (相似度: {predictions[0]['similarity']:.4f})")
    
    else:
        print("无效选择!")

if __name__ == "__main__":
    main()

