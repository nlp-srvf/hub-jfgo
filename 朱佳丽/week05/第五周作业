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
from scipy.spatial.distance import cdist # 导入用于计算距离的函数

#输入模型文件路径
#加载训练好的模型


#思路：
#用Word2Vec将每个标题（句子）转换成一个固定维度的向量。
#用KMeans算法对这些句子向量进行聚类。
#按类别将句子分组并展示。
#现在，我们要实现基于KMeans结果的类内距离排序。这个需求的目的是：找出每个簇（cluster）中，哪些句子最能代表这个簇的中心思想。
#一个句子向量离它所属簇的中心点（centroid）越近，就说明它和这个簇的“平均主题”越相似，因此越具有代表性。所以，我们的任务就是：
#对于每个簇，计算该簇内所有句子向量到该簇中心点的距离。
#根据这个距离，对簇内的句子进行升序排序。
#排序后，排在最前面的句子就是最具代表性的。




def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

def load_sentence(path):
    sentences = set()
    with open(path, encoding="utf8") as f:
        for line in f:
            sentence = line.strip()
            # 注意：这里为了后续能将原始句子和分词后的句子对应，我们最好用list而不是set
            # set会打乱顺序
    # 修改为使用列表来保持原始顺序
    sentences_list = []
    with open(path, encoding="utf8") as f:
        for line in f:
            sentences_list.append(line.strip())
            
    # 分词后的句子，用于向量化，也用列表保持顺序
    cut_sentences = [" ".join(jieba.cut(s)) for s in sentences_list]
    
    print("获取句子数量：", len(sentences_list))
    # 返回原始句子和分词后的句子
    return sentences_list, cut_sentences

#将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  #sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        word_count = 0
        #所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
                word_count += 1
            except KeyError:
                #部分词在训练中未出现，用全0向量代替
                #这里可以忽略不计，因为它们对均值没有贡献
                pass
        # 避免除以零的错误
        if word_count > 0:
            vectors.append(vector / word_count)
        else:
            # 如果句子中所有词都不在词典里,则添加一个零向量
            vectors.append(np.zeros(model.vector_size))
            
    return np.array(vectors)


def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    # load_sentence现在返回原始句子和分词后的句子
    original_sentences, cut_sentences = load_sentence("titles.txt")  
    vectors = sentences_to_vectors(cut_sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(original_sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters, n_init='auto', random_state=42) # 定义一个kmeans计算类, 添加n_init避免warning
    kmeans.fit(vectors)          #进行聚类计算
    
    # 获取所有聚类中心点
    centroids = kmeans.cluster_centers_
    # 获取每个样本所属的聚类标签
    labels = kmeans.labels_
    
    # --- 新增：计算每个点到其所属聚类中心的距离 ---
    # cdist(vectors, centroids) 会计算每个vector到所有centroids的距离
    # 结果是一个 (n_samples, n_clusters) 的矩阵
    distances = cdist(vectors, centroids, 'euclidean')
    
    # 我们只需要每个点到它自己所属聚类的中心的距离
    # 可以用一个巧妙的Numpy索引来获取
    # np.arange(len(labels)) -> [0, 1, 2, ..., n-1]
    # labels -> [cluster_of_0, cluster_of_1, ...]
    # distances[np.arange(len(labels)), labels] 就会取出
    # distance[0, cluster_of_0], distance[1, cluster_of_1], ...
    self_cluster_distances = distances[np.arange(len(labels)), labels]
    
    # --- 新增：按类内距离排序并输出 ---
    sentence_label_dist_dict = defaultdict(list)
    for i in range(len(labels)):
        label = labels[i]
        sentence = original_sentences[i]
        distance = self_cluster_distances[i]
        sentence_label_dist_dict[label].append((sentence, distance))
        
    # 对每个聚类中的句子按距离进行排序
    for label, sent_dist_list in sentence_label_dist_dict.items():
        # 按元组的第二个元素（距离）进行升序排序
        sent_dist_list.sort(key=lambda x: x[1])
        sentence_label_dist_dict[label] = sent_dist_list
        
    # --- 修改后的输出部分 ---
    for label, sorted_sentences in sentence_label_dist_dict.items():
        # 获取聚类中心向量，并找到离中心最近的词作为这个聚类的"关键词"（可选步骤，但很有用）
        center_vector = centroids[label]
        # model.wv.similar_by_vector() 找出与给定向量最相似的词
        try:
            # topn=3 表示找最相似的3个词
            keywords = [word for word, score in model.wv.similar_by_vector(center_vector, topn=3)]
            print(f"cluster {label} (代表词: {', '.join(keywords)}):")
        except Exception as e:
            print(f"cluster {label} (无法找到代表词: {e}):")

        # 打印每个聚类中，距离中心点最近的前10个句子
        for i in range(min(10, len(sorted_sentences))):
            sentence, distance = sorted_sentences[i]
            # 保留两位小数显示距离
            print(f"  {sentence.replace(' ', '')} (distance: {distance:.2f})")
        print("---------")


if __name__ == "__main__":
    main()
