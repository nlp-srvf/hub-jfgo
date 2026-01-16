# coding: utf-8

'''
支持向量机：定义机器学习模型结构
'''

import time
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

class SVMClassifier:
    def __init__(self, config):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            analyzer='char_wb'  # 使用字级特征，对中文有利
        )
        self.model = SVC(
            C=config.svm_c,
            kernel=config.svm_kernel,
            probability=True,  # 启用概率估计
            random_state=config.random_seed
        )

    def preprocess_data(self, texts):
        """文本预处理"""
        # 去除标点符号和特殊字符
        processed_texts = []
        for text in texts:
            # 简单的文本清洗
            text = str(text).strip()
            processed_texts.append(text)
        return processed_texts

    def train(self, X_train, y_train):
        """训练模型"""
        start_time = time.time()
        train_start_time = time.time()
        # 预处理文本
        X_train_processed = self.preprocess_data(X_train)
        # 特征提取（TF-IDF）
        print("Extracting TF-IDF features...")
        X_train_features = self.vectorizer.fit_transform(X_train_processed)
        # 训练模型
        print("Training SVM model...")
        self.model.fit(X_train_features, y_train)
        training_time = time.time() - train_start_time
        print(f'SVM training completed in {training_time:.2f} seconds')
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
        print(f'SVM model saved to {path}')

    def load_model(self, path):
        """加载模型"""
        import pickle
        with open(path, 'rb') as f:
            saved_objects = pickle.load(f)
            self.vectorizer = saved_objects['vectorizer']
            self.model = saved_objects['model']
        print(f'SVM model loaded from {path}')