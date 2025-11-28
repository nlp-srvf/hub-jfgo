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

def sort_sentence_by_intra_cluster_distances(sentences, vectors, labels, cluster_centers):
    '''
    计算类内距离以排序

    参数：
    param sentences: 所有句子
    param vectors: 所有句子的向量
    param labels:  所有句子的簇标签
    param cluster_center: 簇中心

    return:
    sorted_sentence_dict:包含label，sentence，以及distance

    '''

    sentences_list = list(sentences)
    sentences_distances = defaultdict(list)
    for i , (sentence,vector,label) in enumerate(zip(sentences_list, vectors, labels)):
        cluster_center = cluster_centers[label]
        distance = np.linalg.norm(vector - cluster_center)
        sentences_distances[label].append((sentence,distance))
    # 对距离从小到大排序
    sorted_sentence_dict = {}
    for label,sentence_distances in sentences_distances.items():
        sorted_sentences = sorted(sentence_distances, key = lambda x:x[1])
        sorted_sentence_dict[label] = sorted_sentences

    return sorted_sentence_dict





def main():
    model = load_word2vec_model(r"model.w2v") #加载词向量模型
    sentences = load_sentence("titles.txt")  #加载所有标题
    vectors = sentences_to_vectors(sentences, model)   #将所有标题向量化

    n_clusters = int(math.sqrt(len(sentences)))  #指定聚类数量
    print("指定聚类数量：", n_clusters)
    kmeans = KMeans(n_clusters)  #定义一个kmeans计算类
    kmeans.fit(vectors)          #进行聚类计算

    sentence_label_dict = defaultdict(list)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    sorted_sentence_dict = sort_sentence_by_intra_cluster_distances(sentences, vectors, labels, cluster_centers)


    for label, sentences in sorted_sentence_dict.items():
        print("cluster %s :" % label)
        for i in range(min(10, len(sentences))):  #随便打印几个，太多了看不过来
            sentence, distance = sentences[i]
            print(f'距离为{distance:.4f}，{sentence.replace(" ", "")}')
        print("---------")

if __name__ == "__main__":
    main()
