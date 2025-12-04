# 实现基于kmeans结果类内距离的排序
import math
import re
import json
import jieba
import numpy as np
from gensim.models import Word2Vec
from scipy.cluster.vq import kmeans
from sklearn.cluster import KMeans
from collections import defaultdict

# #训练词向量模型
# def train_word2vec_model(corpus_path,dim):
#     sentences = corpus = []
#     with open(corpus_path,encoding='utf-8') as f:
#         for line in f:
#             sentences.append(jieba.lcut(line))
#     model = Word2Vec(corpus,vector_size=dim,sg=1)   #skipgram
#     model.save('model.w2v')
#     print("训练模型已经保存！")
#     return model

#加载词向量模型
def load_word2vec_model(path):
    model = Word2Vec.load(path)
    return model

#加载数据
def load_sentence(path):
    sentences = set()
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sentence=line.strip()
            sentences.add(" ".join(jieba.cut(sentence)))

    print("获取的句子数量：",len(sentences))
    # print(sentences)
    return sentences

#文本向量化
def sentences2vec(sentences,model):
    # print(sentences)
    vectors = []
    for sentence in sentences:
        # print(sentence)
        words = sentence.split()
        # print(words)
        vector=np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        # print(vector)
        vectors.append(vector/len(words))
    # print(vectors)
    return np.array(vectors)

#欧氏距离
def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.square(vector1 - vector2)))

def main():
    #训练模型
    # corpus_path = './corpus.txt'
    # train_model = train_word2vec_model(corpus_path,128)

    model=load_word2vec_model("model.w2v")
    sentences = load_sentence("titles.txt")
    vectors=sentences2vec(sentences,model)
    # print(len(vectors))

    #聚类
    n_clusters =int(math.sqrt(len(sentences)))
    print("指定聚类数量：",n_clusters)
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)

    sentences_label_dict=defaultdict(list)
    for sentence,label in zip(sentences, kmeans.labels_):
            sentences_label_dict[label].append(sentence)
    # print(kmeans.cluster_centers_)

    #计算距离
    per_distance=defaultdict(list)
    #计算每一个向量与其对应的簇心之间的距离
    for index,label in enumerate(kmeans.labels_):
        vector=vectors[index]
        center=kmeans.cluster_centers_[label]
        distance=euclidean_distance(center,vector)
        per_distance[label].append(distance)

    # print(per_distance)
    #求每一类中所有文本到簇心距离的平均值
    for label,distances in per_distance.items():
        per_distance[label]=np.mean(distances)

    #升序排序
    distance_order=sorted(per_distance.items(), key=lambda x:x[1], reverse=False)
    print(distance_order)

    for label, distance in distance_order:
        print("cluster %s :" % label)
        print("distance %f:"%distance)
        sentences=sentences_label_dict[label]
        for i in range(min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()