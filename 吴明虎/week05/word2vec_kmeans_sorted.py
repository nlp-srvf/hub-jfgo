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


def get_two_vector_distance(vector1, vector2):
    # 3种常用向量距离算法
    # distance = 1 - (np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2)))  #Cosine Distance
    distance = np.linalg.norm(vector1 - vector2)   #Euclidean Distance
    # distance = np.sum(np.abs(vector1 - vector2))   #Manhattan Distance
    return distance


def sort_dict_label_by_average(label_dict) -> list:
    #把距离列表求平均并作为label排序依据
    temp_label_dict = {}
    for key, value in label_dict.items():
        temp_label_dict[key] = sum(value) / len(value)
    return sorted(temp_label_dict.items(), key=lambda x: x[1])  # reverse=False


def main():
    model = load_word2vec_model('model.w2v') #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    label_distance_dict = defaultdict(list)
    for sentence, vec, label in zip(sentences, vectors, kmeans.labels_):  #取出句子和标签
        # 使用两个字典，一个存储根据label来的vector，另外一个存储每个词段落的向量与label对应的中心点距离
        sentence_center_distances = get_two_vector_distance(vec,kmeans.cluster_centers_[label])
        label_distance_dict[label].append(sentence_center_distances)
        sentence_label_dict[label].append(sentence)

    # 给label排序
    sorted_label_list=sort_dict_label_by_average(label_distance_dict)

    #排序后按顺序去标签-文本字典里面寻找对应的文本，并打印
    for label in sorted_label_list:
        label_sentences = sentence_label_dict.get(label[0])
        print(" cluster %s : ========= distance : %s " % (label[0], label[1]))
        for i in range(min(10, len(label_sentences))):
            print(label_sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
