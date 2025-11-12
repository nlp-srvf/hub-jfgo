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
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算
    # 计算每个类别的平均欧氏距离
    cluster_distances = []
    for cluster_id in range(n_clusters):
        # 获取当前类别的所有向量
        cluster_indices = np.where(kmeans.labels_ == cluster_id)[0]
        cluster_vectors = vectors[cluster_indices]
        # 获取当前类别的质心
        cluster_center = kmeans.cluster_centers_[cluster_id]
        # 计算每个向量到质心的欧氏距离
        distances = np.linalg.norm(cluster_vectors - cluster_center, axis=1)
        # 计算平均距离
        avg_distance = np.mean(distances)
        cluster_distances.append((cluster_id, avg_distance, cluster_indices))
    # 按平均类内距离从小到大排序
    cluster_distances.sort(key=lambda x: x[1])
    # 按排序后的顺序输出每个类
    for cluster_id, avg_distance, indices in cluster_distances:
        print(f"cluster {cluster_id} (平均欧氏距离: {avg_distance:.4f}):")
        # 输出该类别下的所有句子
        for idx in indices:
            sentence = list(sentences)[idx]
            print(sentence.replace(" ", ""))
        print("---------")

if __name__ == "__main__":
    main()

