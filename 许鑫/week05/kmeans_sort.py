import math
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from collections import defaultdict


# 加载已训练好的模型
def load_word2vec_mode(path):
    model = Word2Vec.load(path)
    return model


def load_sentence(path):
    sentences = set()
    with open(path, encoding='utf-8') as f:
        for line in f:
            sentence = line.strip()
            sentences.add(' '.join(jieba.lcut(sentence)))
    print(f'句子数量： {len(sentences)}')
    return sentences


# 将文本向量化
def sentence_to_vectors(sentences, model):
    vectors = []
    for sentence in sentences:
        words = sentence.split()
        vector = np.zeros(model.vector_size)
        for word in words:
            try:
                vector += model.wv[word]
            except KeyError:
                vector += np.zeros(model.vector_size)
        vectors.append(vector / len(words))
    return np.array(vectors)


def main():
    model = load_word2vec_mode('../model.w2v')
    sentences = load_sentence('../titles.txt')
    vectors = sentence_to_vectors(sentences, model)  # 标题向量化
    n_clusters = int(math.sqrt(len(sentences)))  # 指定聚类数量
    print(f'指定聚类数量： {n_clusters}')
    kmeans = KMeans(n_clusters)
    kmeans.fit(vectors)  # 进行聚类计算
    centers = kmeans.cluster_centers_  # 聚类中心
    distance = []
    for vec, label in zip(vectors, kmeans.labels_):
        center = centers[label]
        dist = np.sqrt(np.sum((vec - center) ** 2))
        distance.append(dist)

    sentence_label_dict = defaultdict(list)
    for sentence, label, dist in zip(sentences, kmeans.labels_, distance):  # 取出句子和标签
        sentence_label_dict[label].append((sentence, dist))
    # 类内排序
    for label, sentence in sentence_label_dict.items():  # 同标签放在一起
        print(f'cluster 类内排序 :{label})')
        sentence_sorted = sorted(sentence, key=lambda x: x[1])
        # input(sentence_sorted)
        for sentence_i, dist in sentence_sorted[:10]:
            print(f'{sentence_i.replace(" ", "")}, --> dist: {dist:4f}')
        print('-' * 30)


if __name__ == '__main__':
    main()


