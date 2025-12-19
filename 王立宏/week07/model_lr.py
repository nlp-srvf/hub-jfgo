# coding: utf-8

'''
朴素贝叶斯：定义机器学习模型结构
'''

import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report

class NaiveBayesClassifier:
    def __init__(self, config):
        self.config = config
        self.vectorizer = CountVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        self.model = MultinomialNB()

    def preprocess_data(self, texts):
        """文本预处理"""
        # 去除标点符号和数字
        processed_texts = []
        for text in texts:
            # 简单的文本清洗
            import re
            text = re.sub(r'[^\u4e00-\u9fa5\s]', '', str(text))
            processed_texts.append(text.strip())
        return processed_texts

    def train(self, X_train, y_train):
        """训练模型"""
        start_time = time.time()

        train_start_time = time.time()

        # 预处理文本
        X_train_processed = self.preprocess_data(X_train)

        # 特征提取
        print("Extracting features...")
        X_train_features = self.vectorizer.fit_transform(X_train_processed)

        # 训练模型
        print("Training Naive Bayes model...")
        self.model.fit(X_train_features, y_train)

        training_time = time.time() - train_start_time
        print(f'Naive Bayes training completed in {training_time:.2f} seconds')

        self.training_time = time.time() - start_time
        return training_time

    def predict(self, texts):
        """预测"""
        start_time = time.time()

        # 预处理文本
        texts_processed = self.preprocess_data(texts)

        # 特征提取
        X_features = self.vectorizer.transform(texts_processed)

        # 预测
        predictions = self.model.predict(X_features)
        probabilities = self.model.predict_proba(X_features)

        inference_time = time.time() - start_time
        return predictions, probabilities, inference_time

    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions, probabilities, inference_time = self.predict(X_test)

        # 计算指标
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)

        return accuracy, report, inference_time

    def save_model(self, path):
        """保存模型"""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'model': self.model
            }, f)
        print(f'Naive Bayes model saved to {path}')

    def load_model(self, path):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            saved_objects = pickle.load(f)
            self.vectorizer = saved_objects['vectorizer']
            self.model = saved_objects['model']
        print(f'Naive Bayes model loaded from {path}')