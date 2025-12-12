# -*- coding: utf-8 -*-

import torch
import re
import json
import numpy as np
from collections import defaultdict
from loader import load_data
from config import Config
from model import TripletNetwork

"""
模型效果测试 - 使用三元组损失训练的模型进行预测
"""

class SentenceMatchPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.model = TripletNetwork(config)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 加载训练数据作为知识库
        train_data = load_data(config["train_data_path"], config, shuffle=False)
        self.train_data_dataset = train_data.dataset
        self.schema = self.train_data_dataset.schema
        
        # 构建反向schema映射
        self.index_to_label = dict((y, x) for x, y in self.schema.items())
        
        # 将知识库向量化
        self.knwb_to_vector()
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        print("模型加载完成!")
    
    def knwb_to_vector(self):
        """将知识库中的问题向量化"""
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        
        for standard_question_index, question_ids in self.train_data_dataset.knwb.items():
            for question_id in question_ids:
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            # 归一化向量
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        
        print(f"知识库向量化完成，共{len(self.question_ids)}个问题")
    
    def encode_sentence(self, text):
        """将句子编码成ID序列"""
        input_id = []
        for char in text:
            input_id.append(self.train_data_dataset.vocab.get(char, self.train_data_dataset.vocab.get("[UNK]", 0)))
        # padding
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id
    
    def predict(self, sentence, top_k=5):
        """
        预测句子的类别和最相似的问题
        :param sentence: 输入句子
        :param top_k: 返回top k个最相似的问题
        :return: 预测的类别和最相似的问题列表
        """
        # 编码输入句子
        input_id = self.encode_sentence(sentence)
        input_tensor = torch.LongTensor([input_id])
        
        if torch.cuda.is_available():
            input_tensor = input_tensor.cuda()
        
        # 获取句子向量
        with torch.no_grad():
            sentence_vector = self.model(input_tensor)
            sentence_vector = torch.nn.functional.normalize(sentence_vector, dim=-1)
        
        # 计算与知识库中所有问题的相似度
        res = torch.mm(sentence_vector, self.knwb_vectors.T)
        similarities = res.squeeze().cpu().numpy()
        
        # 获取top k个最相似的问题
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        # 获取预测类别（最相似问题的类别）
        hit_index = top_k_indices[0]
        predicted_label_index = self.question_index_to_standard_question_index[hit_index]
        predicted_label = self.index_to_label[predicted_label_index]
        
        # 构建返回结果
        similar_questions = []
        for idx in top_k_indices:
            question_id = self.question_ids[idx]
            label_index = self.question_index_to_standard_question_index[idx]
            label = self.index_to_label[label_index]
            similarity = float(similarities[idx])
            similar_questions.append({
                "similarity": similarity,
                "label": label
            })
        
        return predicted_label, similar_questions


def main():
    # 加载配置
    model_path = "model_output/epoch_10.pth"
    
    # 创建预测器
    predictor = SentenceMatchPredictor(Config, model_path)
    
    # 测试样例
    test_sentences = [
        "我想查一下我的话费",
        "帮我看看积分有多少",
        "手机丢了怎么办",
        "怎么修改密码",
        "开通来电显示多少钱"
    ]
    
    print("\n" + "="*50)
    print("三元组损失模型预测测试")
    print("="*50)
    
    for sentence in test_sentences:
        print(f"\n输入: {sentence}")
        predicted_label, similar_questions = predictor.predict(sentence, top_k=3)
        print(f"预测类别: {predicted_label}")
        print(f"最相似的3个问题:")
        for i, item in enumerate(similar_questions, 1):
            print(f"  {i}. 类别: {item['label']}, 相似度: {item['similarity']:.4f}")


if __name__ == "__main__":
    main()

