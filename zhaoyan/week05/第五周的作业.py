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
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import defaultdict
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei'] if 'SimHei' in plt.rcParams['font.sans-serif'] else ['Microsoft YaHei',
                                                                                                  'KaiTi', 'STKaiti',
                                                                                                  'STXihei', 'YouYuan']
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

# 计算两个向量之间的欧氏距离
def calculate_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

# 冒泡排序函数
def bubble_sort(items):
    n = len(items)
    for i in range(n):
        for j in range(0, n-i-1):
            if items[j][1] > items[j+1][1]:  # 按平均距离排序
                items[j], items[j+1] = items[j+1], items[j]
    return items
def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    for sentence, label in zip(sentences, kmeans.labels_):  #取出句子和标签
        #kmeans.cluester_centers_ #每个聚类中心
        sentence_label_dict[label].append(sentence)         #同标签的放到一起
    # for label, sentences in sentence_label_dict.items():
    #     print("cluster %s :" % label)
    #     for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
    #         print(sentences[i].replace(" ", ""))
    #     print("---------")


    cluster_distances = []
    for label, sentences in sentence_label_dict.items():
        # center = kmeans.cluster_centers[label]
        center = kmeans.cluster_centers_[label]  # 修正属性名
        total_distance = 0
        print("cluster %s :" % label)
        # 获取该聚类中所有句子对应的索引
        cluster_indices = [i for i, lbl in enumerate(kmeans.labels_) if lbl == label]

        print("cluster %s :" % label)
        # 使用前面已经计算好的向量
        for idx in cluster_indices:
            sentence_vector = vectors[idx]
            distance = calculate_distance(sentence_vector, center)
            total_distance += distance
            # 计算平均距离
        avg_distance = total_distance / len(sentences)
        cluster_distances.append((label, avg_distance))
        print("平均距离: %.4f" % avg_distance)
        print("---------")

    # 使用冒泡排序
    print("\n=== 按平均距离排序 ===")
    sorted_clusters = bubble_sort(cluster_distances)

    # sentence_label_dict 是一个 defaultdict
    # 键是聚类标签，值是
    # list
    # 示例：{0: ['句子1', '句子2'], 1: ['句子3', '句子4']}

    top_10_clusters = sorted_clusters[:10]
    for label, avg_distance in top_10_clusters:
        print("聚类 %d: 平均距离 = %.4f" % (label, avg_distance))
        print(f"包含 {len(sentence_label_dict[label])} 个句子")

        sentences_in_cluster = sentence_label_dict[label]

        for i in range(min(10, len(sentences_in_cluster))):  # 随便打印几个，太多了看不过来
            print(sentences_in_cluster[i].replace(" ", ""))
        print("---------")

    # 绘制平均距离曲线图
    plt.figure(figsize=(12, 6))

    # 提取前10个聚类的标签和距离
    labels = [f"聚类{label}" for label, _ in top_10_clusters]
    distances = [distance for _, distance in top_10_clusters]

    # 创建曲线图
    plt.plot(labels, distances, marker='o', linewidth=2, markersize=8)
    plt.title('前10个聚类的平均距离曲线图', fontsize=14)
    plt.xlabel('聚类', fontsize=12)
    plt.ylabel('平均距离', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 在点上添加数值标签
    for i, v in enumerate(distances):
        plt.text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)

    plt.xticks(rotation=45)  # 旋转x轴标签，避免重叠
    plt.tight_layout()  # 自动调整布局
    plt.show()


if __name__ == "__main__":
    main()

