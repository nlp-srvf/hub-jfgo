# coding: utf-8

'''
predict.py：模型效果测试
'''

import torch
import json
import os
from typing import List, Tuple, Dict, Any, Optional
import argparse

from config import Config
from device_utils import set_device
from loader import NERDataLoader
from model import create_model
from transformers import BertTokenizerFast


class NERPredictor:
    """
    NER预测器
    用于对输入文本进行命名实体识别预测
    """
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        """
        初始化预测器
        
        Args:
            model_path: 训练好的模型路径
            config: 配置字典
        """
        self.config = config
        self.device = set_device()
        
        # 加载标签映射
        self.data_loader = NERDataLoader(config)
        self.label2id, self.id2label = self.data_loader.get_label_mappings()
        
        # 加载分词器
        self.tokenizer = BertTokenizerFast.from_pretrained(config['bert_path'])
        
        # 加载模型
        self.model = create_model(config)
        self.model.to(self.device)
        self.load_model(model_path)
        
        # 设置模型为评估模式
        self.model.eval()
        
        print(f"✅ 预测器初始化完成")
        print(f"🏷️  标签数量: {len(self.id2label)}")
        print(f"📋 标签列表: {list(self.id2label.values())}")
    
    def load_model(self, model_path: str):
        """
        加载训练好的模型
        
        Args:
            model_path: 模型文件路径
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"📦 模型已从 {model_path} 加载")
        print(f"📊 训练信息 - Epoch: {checkpoint['epoch']}, Step: {checkpoint['global_step']}")
        
        if 'metrics' in checkpoint:
            metrics = checkpoint['metrics']
            print(f"🎯 模型性能 - F1: {metrics.get('overall_f1', 0):.4f}, "
                  f"Precision: {metrics.get('overall_precision', 0):.4f}, "
                  f"Recall: {metrics.get('overall_recall', 0):.4f}")
    
    def preprocess_text(self, text: str) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """
        预处理输入文本
        
        Args:
            text: 输入文本
            
        Returns:
            tokens: 分词后的token列表
            encoded: 编码后的tensor字典
        """
        # 对于中文，按字符分割
        tokens = list(text)
        
        # 使用BERT分词器编码
        encoded = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.config['max_length'],
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return tokens, encoded
    
    def predict_single(self, text: str) -> Dict[str, Any]:
        """
        对单个文本进行预测
        
        Args:
            text: 输入文本
            
        Returns:
            预测结果字典
        """
        # 预处理
        tokens, encoded = self.preprocess_text(text)
        
        # 将数据移到设备
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        token_type_ids = encoded['token_type_ids'].to(self.device)
        
        # 模型预测
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            
            # 获取预测标签
            predictions = self.model.decode(outputs['emissions'], outputs['mask'])
        
        # 处理预测结果
        predicted_labels = predictions[0]  # 取第一个（也是唯一一个）样本
        
        # 将预测结果与原始tokens对齐
        word_ids = encoded.word_ids()
        aligned_predictions = []
        aligned_tokens = []
        
        previous_word_idx = None
        for i, word_idx in enumerate(word_ids):
            if word_idx is None:
                continue  # 跳过特殊token
            elif word_idx != previous_word_idx:
                if word_idx < len(predicted_labels):
                    label_id = predicted_labels[word_idx]
                    label_name = self.id2label[label_id]
                    aligned_predictions.append(label_name)
                    aligned_tokens.append(tokens[word_idx])
                    previous_word_idx = word_idx
        
        # 提取命名实体
        entities = self._extract_entities(aligned_tokens, aligned_predictions)
        
        return {
            'text': text,
            'tokens': aligned_tokens,
            'predictions': aligned_predictions,
            'entities': entities
        }
    
    def _extract_entities(self, tokens: List[str], labels: List[str]) -> List[Dict[str, Any]]:
        """
        从预测标签中提取命名实体
        
        Args:
            tokens: token列表
            labels: 对应的标签列表
            
        Returns:
            提取的实体列表
        """
        entities = []
        current_entity = None
        
        for i, (token, label) in enumerate(zip(tokens, labels)):
            if label.startswith('B-'):
                # 开始新的实体
                if current_entity:
                    entities.append(current_entity)
                
                entity_type = label[2:]  # 去掉'B-'
                current_entity = {
                    'text': token,
                    'label': entity_type,
                    'start': i,
                    'end': i,
                    'tokens': [token],
                    'type': entity_type
                }
            
            elif label.startswith('I-'):
                # 继续当前实体
                entity_type = label[2:]  # 去掉'I-'
                if current_entity and current_entity['type'] == entity_type:
                    # 继续当前实体
                    current_entity['text'] += token
                    current_entity['end'] = i
                    current_entity['tokens'].append(token)
                else:
                    # 开始新实体（错误的标签序列，但我们还是开始新实体）
                    if current_entity:
                        entities.append(current_entity)
                    current_entity = {
                        'text': token,
                        'label': entity_type,
                        'start': i,
                        'end': i,
                        'tokens': [token],
                        'type': entity_type
                    }
            
            else:  # O标签
                # 结束当前实体
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None
        
        # 处理最后一个实体
        if current_entity:
            entities.append(current_entity)
        
        return entities
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            texts: 输入文本列表
            
        Returns:
            预测结果列表
        """
        results = []
        
        print(f"🔄 开始批量预测 {len(texts)} 个文本...")
        
        for i, text in enumerate(texts):
            result = self.predict_single(text)
            results.append(result)
            
            # 打印进度
            if (i + 1) % 10 == 0 or i == len(texts) - 1:
                print(f"📊 已处理 {i + 1}/{len(texts)} 个文本")
        
        return results
    
    def print_result(self, result: Dict[str, Any]):
        """
        打印预测结果
        
        Args:
            result: 预测结果字典
        """
        print(f"\n📝 原始文本: {result['text']}")
        print(f"🏷️  标签序列: {' '.join(result['predictions'])}")
        
        if result['entities']:
            print(f"🎯 识别的实体 ({len(result['entities'])}个):")
            for i, entity in enumerate(result['entities'], 1):
                print(f"  {i}. {entity['text']} [{entity['type']}] "
                      f"(位置: {entity['start']}-{entity['end']})")
        else:
            print("🎯 未识别到实体")
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """
        保存预测结果
        
        Args:
            results: 预测结果列表
            output_path: 输出文件路径
        """
        # 准备保存的数据
        save_data = []
        for result in results:
            save_data.append({
                'text': result['text'],
                'entities': result['entities'],
                'tokens': result['tokens'],
                'predictions': result['predictions']
            })
        
        # 保存为JSON格式
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"💾 预测结果已保存到: {output_path}")
    
    def interactive_predict(self):
        """
        交互式预测
        """
        print("\n" + "="*60)
        print("🎯 NER 交互式预测")
        print("="*60)
        print("输入文本进行命名实体识别，输入 'quit' 或 'exit' 退出")
        print("-"*60)
        
        while True:
            try:
                text = input("\n📝 请输入文本: ").strip()
                
                if text.lower() in ['quit', 'exit', 'q', '退出']:
                    print("👋 再见！")
                    break
                
                if not text:
                    print("⚠️  请输入有效文本")
                    continue
                
                # 预测
                result = self.predict_single(text)
                
                # 显示结果
                self.print_result(result)
                
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 预测出错: {e}")


def test_predictor():
    """
    测试预测器功能
    """
    print("🧪 测试NER预测器...")
    
    # 检查模型是否存在
    model_path = os.path.join(Config['model_path'], 'best_model.pt')
    if not os.path.exists(model_path):
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先训练模型或提供正确的模型路径")
        return
    
    # 创建预测器
    predictor = NERPredictor(model_path, Config)
    
    # 测试文本
    test_texts = [
        "张三在北京的清华大学工作",
        "中国政府代表团于昨天访问了美国纽约",
        "李四是一家科技公司的创始人"
    ]
    
    print(f"\n📋 测试文本 ({len(test_texts)}个):")
    for i, text in enumerate(test_texts, 1):
        print(f"{i}. {text}")
    
    # 批量预测
    results = predictor.predict_batch(test_texts)
    
    # 显示结果
    print(f"\n🎯 预测结果:")
    for i, result in enumerate(results, 1):
        predictor.print_result(result)
        print()
    
    # 保存结果
    output_path = os.path.join(Config['model_path'], 'test_predictions.json')
    predictor.save_results(results, output_path)
    
    print("✅ 预测器测试完成！")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='BERT+CRF NER模型预测')
    
    parser.add_argument('--model_path', type=str, 
                       default=os.path.join(Config['model_path'], 'best_model.pt'),
                       help='模型文件路径')
    parser.add_argument('--text', type=str, help='要预测的文本')
    parser.add_argument('--input_file', type=str, help='输入文本文件路径')
    parser.add_argument('--output_file', type=str, help='输出结果文件路径')
    parser.add_argument('--interactive', action='store_true', help='交互式预测模式')
    parser.add_argument('--test', action='store_true', help='测试预测器功能')
    
    args = parser.parse_args()
    
    # 测试模式
    if args.test:
        test_predictor()
        return
    
    # 创建预测器
    try:
        predictor = NERPredictor(args.model_path, Config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 交互式预测
    if args.interactive:
        predictor.interactive_predict()
        return
    
    # 单文本预测
    if args.text:
        result = predictor.predict_single(args.text)
        predictor.print_result(result)
        
        if args.output_file:
            predictor.save_results([result], args.output_file)
        return
    
    # 文件批量预测
    if args.input_file:
        if not os.path.exists(args.input_file):
            print(f"❌ 输入文件不存在: {args.input_file}")
            return
        
        # 读取文本文件
        with open(args.input_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        print(f"📖 从文件读取了 {len(texts)} 个文本")
        
        # 批量预测
        results = predictor.predict_batch(texts)
        
        # 保存结果
        if args.output_file:
            predictor.save_results(results, args.output_file)
        else:
            # 显示前几个结果
            for i, result in enumerate(results[:3], 1):
                predictor.print_result(result)
        
        return
    
    # 默认交互模式
    print("未指定预测内容，进入交互模式...")
    predictor.interactive_predict()


if __name__ == "__main__":
    main()

