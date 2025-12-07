# coding: utf-8

import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict

'''
实现基于kmeans结果类内距离的排序
'''


def load_word2vec_model(model_path):
    '''
    加载预训练的Word2Vec词向量模型
    :param model_path: 模型文件路径
    :return: Word2Vec模型对象
    '''
    model = Word2Vec.load(model_path)
    return model


def load_sentence(file_path):
    '''
    从文件加载句子并进行分词处理
    :param file_path: 文本文件路径
    :return: 分词后的句子集合
    '''
    sentences = set()
    with open(file_path, encoding="utf8") as f:  # 以UTF-8编码打开文件
        for line in f:  # 按行获取文件内容
            sentence = line.strip()  # 去除行首尾的空白字符
            sentences.add(" ".join(jieba.cut(sentence)))
    return sentences


# 将文本向量化
def sentences_to_vectors(sentences, model):
    '''
    将句子转换为向量
    :param sentences: 分词后的句子集合
    :param model: Word2Vec模型
    :return: 句子向量数组
    '''
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # 将通过空格进行分词后的句子按空格拆分为单词列表
        vector = np.zeros(model.vector_size)  # 初始化与词向量维度相同的零向量
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                # 将词的向量加到句子向量中
                vector += model.wv[word]
            except KeyError:
                # 训练中未出现的词，使用全0向量代替
                vector += np.zeros(model.vector_size)
        # 计算平均向量并添加到结果列表
        vectors.append(vector / len(words))
    return np.array(vectors)


def main(model_path, file_path):
    # 加载预训练的词向量模型
    model = load_word2vec_model(model_path)
    # 加载并预处理文本数据
    sentences = load_sentence(file_path)
    # 将文本转换为向量
    vectors = sentences_to_vectors(sentences, model)
    # 计算聚类数量（使用句子数量的平方根）
    n_clusters = int(math.sqrt(len(sentences)))
    print("聚类数量：", n_clusters)
    # 初始化 K-means 聚类器
    kmeans = KMeans(n_clusters)
    # 对句子向量进行聚类计算
    kmeans.fit(vectors)

    # 创建字典来存储聚类结果（使用默认字典，值为列表类型）
    sentence_label_dict = defaultdict(list)

    # 组织聚类结果并计算距离
    for sentence, label, vec in zip(sentences, kmeans.labels_, vectors):  # 同时遍历句子、聚类标签和向量
        # 获取当前标签对应的聚类中心向量
        center = kmeans.cluster_centers_[label]
        # 计算当前句子向量到聚类中心（中间点）的欧氏距离
        distance = np.sqrt(np.sum(vec - center) ** 2)
        # 累加距离，并将距离存放在每组的数组第一位
        if label not in sentence_label_dict:  # 如果是该标签的第一个句子
            # 初始化距离累加值
            sentence_label_dict[label].append(distance)
        else:
            # 累加距离值
            sentence_label_dict[label][0] += distance
        # 将标签相同的句子放在一起
        sentence_label_dict[label].append(sentence)

    # 计算每个聚类的平均距离
    for k, v in sentence_label_dict.items():
        # 平均距离 = 总距离 / (句子数量 - 1)，因为第一个元素是距离和
        sentence_label_dict[k][0] = v[0] / (len(v) - 1)
    # 根据平均距离对聚类进行排序（从紧密到松散，平均距离越小，聚类越紧密）
    sort_dict = dict(sorted(sentence_label_dict.items(), key=lambda x: x[1][0]))
    # 输出聚类结果
    for label, sentences in sort_dict.items():
        print("cluster-", label, "（平均距离 = ", sentences[0], "）")
        # 打印每个聚类的前10个句子（最多10个）
        for i in range(1, min(11, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("=====================================================\n")


if __name__ == "__main__":
    model_path = "model.w2v"
    file_path = "titles.txt"
    main(model_path, file_path)
