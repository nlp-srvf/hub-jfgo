# -*- coding: utf-8 -*-

import torch
import re
import numpy as np
from collections import defaultdict
from loader import load_data

"""
模型效果测试 - BERT版本
"""


class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)

    def eval(self, epoch):
        """
        评估模型效果
        :param epoch: 当前轮数
        :return: Micro-F1分数
        """
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {
            "LOCATION": defaultdict(int),
            "TIME": defaultdict(int),
            "PERSON": defaultdict(int),
            "ORGANIZATION": defaultdict(int)
        }
        self.model.eval()
        
        for index, batch_data in enumerate(self.valid_data):
            # 获取当前batch对应的原始句子
            sentences = self.valid_data.dataset.sentences[
                index * self.config["batch_size"]: (index + 1) * self.config["batch_size"]
            ]
            
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            
            # BERT模型需要input_ids, attention_mask, labels
            input_ids, attention_mask, labels = batch_data
            
            with torch.no_grad():
                # 不输入labels，使用模型当前参数进行预测
                pred_results = self.model(input_ids, attention_mask)
            
            self.write_stats(labels, pred_results, sentences)
        
        return self.show_stats()

    def write_stats(self, labels, pred_results, sentences):
        """
        统计预测结果
        """
        assert len(labels) == len(pred_results) == len(sentences)
        
        if not self.config["use_crf"]:
            pred_results = torch.argmax(pred_results, dim=-1)
        
        for true_label, pred_label, sentence in zip(labels, pred_results, sentences):
            if not self.config["use_crf"]:
                pred_label = pred_label.cpu().detach().tolist()
            true_label = true_label.cpu().detach().tolist()
            
            # 对于BERT，需要去掉[CLS]和[SEP]对应的标签
            # 标签中-1表示特殊token或padding，需要跳过
            # 从索引1开始（跳过[CLS]），取句子长度个标签
            true_label_clean = true_label[1:len(sentence)+1]
            pred_label_clean = pred_label[1:len(sentence)+1]
            
            # 解码实体
            true_entities = self.decode(sentence, true_label_clean)
            pred_entities = self.decode(sentence, pred_label_clean)
            
            # 统计各类实体
            for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
                self.stats_dict[key]["正确识别"] += len(
                    [ent for ent in pred_entities[key] if ent in true_entities[key]]
                )
                self.stats_dict[key]["样本实体数"] += len(true_entities[key])
                self.stats_dict[key]["识别出实体数"] += len(pred_entities[key])
        return

    def show_stats(self):
        """
        显示评估结果
        :return: Micro-F1分数
        """
        F1_scores = []
        for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]:
            # 正确率 = 识别出的正确实体数 / 识别出的实体数
            # 召回率 = 识别出的正确实体数 / 样本的实体数
            precision = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["识别出实体数"])
            recall = self.stats_dict[key]["正确识别"] / (1e-5 + self.stats_dict[key]["样本实体数"])
            F1 = (2 * precision * recall) / (precision + recall + 1e-5)
            F1_scores.append(F1)
            self.logger.info("%s类实体，准确率：%f, 召回率: %f, F1: %f" % (key, precision, recall, F1))
        
        self.logger.info("Macro-F1: %f" % np.mean(F1_scores))
        
        # 计算Micro-F1
        correct_pred = sum([self.stats_dict[key]["正确识别"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        total_pred = sum([self.stats_dict[key]["识别出实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        true_enti = sum([self.stats_dict[key]["样本实体数"] for key in ["PERSON", "LOCATION", "TIME", "ORGANIZATION"]])
        
        micro_precision = correct_pred / (total_pred + 1e-5)
        micro_recall = correct_pred / (true_enti + 1e-5)
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall + 1e-5)
        
        self.logger.info("Micro-F1: %f" % micro_f1)
        self.logger.info("--------------------")
        
        return micro_f1

    def decode(self, sentence, labels):
        """
        解码预测结果，提取实体
        
        标签schema:
        {
          "B-LOCATION": 0,
          "B-ORGANIZATION": 1,
          "B-PERSON": 2,
          "B-TIME": 3,
          "I-LOCATION": 4,
          "I-ORGANIZATION": 5,
          "I-PERSON": 6,
          "I-TIME": 7,
          "O": 8
        }
        """
        labels = "".join([str(x) for x in labels[:len(sentence)]])
        results = defaultdict(list)
        
        # 使用正则表达式提取实体
        # B-LOCATION(0) + I-LOCATION(4)*
        for location in re.finditer("(04+)", labels):
            s, e = location.span()
            results["LOCATION"].append(sentence[s:e])
        
        # B-ORGANIZATION(1) + I-ORGANIZATION(5)*
        for location in re.finditer("(15+)", labels):
            s, e = location.span()
            results["ORGANIZATION"].append(sentence[s:e])
        
        # B-PERSON(2) + I-PERSON(6)*
        for location in re.finditer("(26+)", labels):
            s, e = location.span()
            results["PERSON"].append(sentence[s:e])
        
        # B-TIME(3) + I-TIME(7)*
        for location in re.finditer("(37+)", labels):
            s, e = location.span()
            results["TIME"].append(sentence[s:e])
        
        return results

