# 基于训练好的词向量模型进行聚类
# 聚类采用Kmeans算法
import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 输入模型文件路径
# 加载训练好的模型
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


# 将文本向量化
def sentences_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()  # sentence是分好词的，空格分开
        vector = np.zeros(model.vector_size)
        # 所有词的向量相加求平均，作为句子向量
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                # 部分词在训练中未出现，用全0向量代替
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

    sentence_label_dict = defaultdict(list)

    for sentence, label, vec in zip(sentences, kmeans.labels_, vectors):  # 取出句子、标签、标题文本向量
        # 获取当前label对应的中心点向量
        center = kmeans.cluster_centers_[label]
        # 计算当前标题文本向量与中间点向量距离（欧氏距离）
        distance = np.sqrt(np.sum(vec - center) ** 2)
        # 累加距离,并将距离存放每组的数组中第一位
        if label not in sentence_label_dict:
            sentence_label_dict[label].append(distance)
        else:
            sentence_label_dict[label][0] += distance
        # 同标签的放到一起
        sentence_label_dict[label].append(sentence)

    # 计算平均值
    for k, v in sentence_label_dict.items():
        sentence_label_dict[k][0] = v[0] / (len(v) - 1)
    # 根据距离平均值排序
    sort_dict = dict(sorted(sentence_label_dict.items(), key=lambda x: x[1][0]))

    for label, sentences in sort_dict.items():
        print("cluster %s （distance_avg = %s）:" % (label, sentences[0]))
        for i in range(1, min(10, len(sentences))):
            print(sentences[i].replace(" ", ""))
        print("---------")


if __name__ == "__main__":
    main()
