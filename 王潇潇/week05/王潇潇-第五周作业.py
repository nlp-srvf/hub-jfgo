#!/usr/bin/env python3
#coding: utf-8

#基于训练好的词向量模型进行聚类
#聚类采用Kmeans算法
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

#输入模型文件路径
#加载训练好的模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))
    print("获取句子数量：", len(sentences))
    return sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v")  # 加载词向量模型
    sentences = load_sentence("titles.txt")  # 加载所有标题
    vectors = sentences_to_vectors(sentences, model)  # 将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  # 定义一个kmeans计算类
    kmeans.fit(vectors)  # 进行聚类计算

    sentence_info = list(zip(sentences, kmeans.labels_, vectors))

    # 按标签分组，并计算每个类的平均距离
    cluster_dict = defaultdict(list)
    for sent, label, vec in sentence_info:
        cluster_dict[label].append((sent, vec))  # 存储句子和对应的向量

    # 计算每个类的平均距离
    cluster_avg_dist = []
    for label, items in cluster_dict.items():
        center = kmeans.cluster_centers_[label]  # 当前类的聚类中心
        total_dist = 0.0
        for sent, vec in items:
            # 计算向量与中心的距离
            dist = np.linalg.norm(vec - center)
            # 累加
            total_dist += dist
        # 求平均
        avg_dist = total_dist / len(items)  
        cluster_avg_dist.append((label, avg_dist))

    # 排序输出
    cluster_avg_dist.sort(key=lambda x: x[1])

    # 按排序后的标签输出结果
    for label, avg_dist in cluster_avg_dist:
        items = cluster_dict[label]
        print(f"cluster {label} (平均距离: {avg_dist:.4f}):")  # 打印平均距离
        # 打印该类的前10个句子
        for i in range(min(10, len(items))):
            print(items[i][0].replace(" ", ""))  # 去掉分词的空格
        print("---------")

if __name__ == "__main__":
    main()

