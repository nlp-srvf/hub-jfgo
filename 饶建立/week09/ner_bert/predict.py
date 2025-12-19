# -*- coding: utf-8 -*-

import torch
import re
from collections import defaultdict
from transformers import BertTokenizer
from model import BertNERModel
from config import Config

"""
BERT NER模型预测
"""


class NERPredictor:
    def __init__(self, config, model_path):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        
        # 加载模型
        self.model = BertNERModel(config)
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()
        
        # GPU
        self.cuda_flag = torch.cuda.is_available()
        if self.cuda_flag:
            self.model = self.model.cuda()
        
        # 标签映射
        self.id2label = {
            0: "B-LOCATION",
            1: "B-ORGANIZATION",
            2: "B-PERSON",
            3: "B-TIME",
            4: "I-LOCATION",
            5: "I-ORGANIZATION",
            6: "I-PERSON",
            7: "I-TIME",
            8: "O"
        }
        
        print("模型加载完成!")

    def predict(self, sentence):
        """
        对单个句子进行NER预测
        :param sentence: 输入句子（字符串）
        :return: 识别出的实体字典
        """
        # 将句子转换为字符列表
        chars = list(sentence)
        max_length = self.config["max_length"]
        
        # 截断
        max_char_length = max_length - 2
        chars = chars[:max_char_length]
        
        # 构建输入
        tokens = ["[CLS]"] + chars + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        attention_mask = [1] * len(input_ids)
        
        # Padding
        padding_length = max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * padding_length
        attention_mask += [0] * padding_length
        
        # 转换为tensor
        input_ids = torch.LongTensor([input_ids])
        attention_mask = torch.LongTensor([attention_mask])
        
        if self.cuda_flag:
            input_ids = input_ids.cuda()
            attention_mask = attention_mask.cuda()
        
        # 预测
        with torch.no_grad():
            pred_results = self.model(input_ids, attention_mask)
        
        # 处理预测结果
        if self.config["use_crf"]:
            pred_labels = pred_results[0]  # CRF返回list
        else:
            pred_labels = torch.argmax(pred_results, dim=-1)[0].cpu().tolist()
        
        # 去掉[CLS]和[SEP]对应的标签，取句子长度个标签
        pred_labels = pred_labels[1:len(chars)+1]
        
        # 解码实体
        entities = self.decode(sentence[:len(chars)], pred_labels)
        
        return entities

    def decode(self, sentence, labels):
        """
        解码预测结果，提取实体
        """
        labels_str = "".join([str(x) for x in labels])
        results = defaultdict(list)
        
        # B-LOCATION(0) + I-LOCATION(4)*
        for match in re.finditer("(04*)", labels_str):
            s, e = match.span()
            results["LOCATION"].append({
                "entity": sentence[s:e],
                "start": s,
                "end": e
            })
        
        # B-ORGANIZATION(1) + I-ORGANIZATION(5)*
        for match in re.finditer("(15*)", labels_str):
            s, e = match.span()
            results["ORGANIZATION"].append({
                "entity": sentence[s:e],
                "start": s,
                "end": e
            })
        
        # B-PERSON(2) + I-PERSON(6)*
        for match in re.finditer("(26*)", labels_str):
            s, e = match.span()
            results["PERSON"].append({
                "entity": sentence[s:e],
                "start": s,
                "end": e
            })
        
        # B-TIME(3) + I-TIME(7)*
        for match in re.finditer("(37*)", labels_str):
            s, e = match.span()
            results["TIME"].append({
                "entity": sentence[s:e],
                "start": s,
                "end": e
            })
        
        return results

    def get_labeled_sentence(self, sentence):
        """
        返回带有实体标注的句子
        """
        entities = self.predict(sentence)
        result = []
        
        # 收集所有实体及其位置
        all_entities = []
        for ent_type, ent_list in entities.items():
            for ent in ent_list:
                all_entities.append({
                    "type": ent_type,
                    "entity": ent["entity"],
                    "start": ent["start"],
                    "end": ent["end"]
                })
        
        # 按位置排序
        all_entities.sort(key=lambda x: x["start"])
        
        # 构建标注结果
        last_end = 0
        for ent in all_entities:
            if ent["start"] >= last_end:
                result.append(sentence[last_end:ent["start"]])
                result.append(f"[{ent['entity']}/{ent['type']}]")
                last_end = ent["end"]
        result.append(sentence[last_end:])
        
        return "".join(result)


def main():
    model_path = "model_output/best_model.pth"
    
    # 创建预测器
    predictor = NERPredictor(Config, model_path)
    
    # 测试样例
    test_sentences = [
        "中国国家主席习近平在北京人民大会堂会见了来访的美国总统拜登。",
        "2023年10月1日，张三在上海参加了阿里巴巴举办的技术大会。",
        "李明昨天去了故宫博物院参观。",
        "华为公司在深圳发布了新款手机。"
    ]
    
    print("\n" + "=" * 60)
    print("BERT NER 模型预测测试")
    print("=" * 60)
    
    for sentence in test_sentences:
        print(f"\n原句: {sentence}")
        entities = predictor.predict(sentence)
        print("识别的实体:")
        for ent_type, ent_list in entities.items():
            if ent_list:
                print(f"  {ent_type}: {[e['entity'] for e in ent_list]}")
        
        labeled = predictor.get_labeled_sentence(sentence)
        print(f"标注结果: {labeled}")


if __name__ == "__main__":
    main()

