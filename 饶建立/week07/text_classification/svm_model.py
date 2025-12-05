# -*- coding: utf-8 -*-
"""
SVM文本分类模型
"""
import time
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from config import SVM_CONFIG, MODEL_DIR
import os


class SVMClassifier:
    def __init__(self, config=None):
        self.config = config or SVM_CONFIG
        self.vectorizer = TfidfVectorizer(
            max_features=self.config['max_features'],
            ngram_range=(1, 2)
        )
        self.model = SVC(
            C=self.config['C'],
            kernel=self.config['kernel'],
            random_state=42
        )
        self.learning_rate = "N/A"  # SVM没有学习率
    
    def train(self, X_train, y_train, X_valid, y_valid):
        """训练模型"""
        print("\n" + "=" * 50)
        print("训练SVM模型...")
        print("=" * 50)
        
        # TF-IDF特征提取
        print("提取TF-IDF特征...")
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        X_valid_tfidf = self.vectorizer.transform(X_valid)
        
        # 训练
        print("训练SVM分类器...")
        start_time = time.time()
        self.model.fit(X_train_tfidf, y_train)
        train_time = time.time() - start_time
        print(f"训练耗时: {train_time:.2f}秒")
        
        # 验证集评估
        y_valid_pred = self.model.predict(X_valid_tfidf)
        valid_acc = accuracy_score(y_valid, y_valid_pred)
        print(f"验证集准确率: {valid_acc:.4f}")
        
        return valid_acc
    
    def evaluate(self, X_test, y_test):
        """评估模型"""
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"\n测试集准确率: {acc:.4f}")
        print("\n分类报告:")
        print(classification_report(y_test, y_pred, target_names=['差评', '好评']))
        
        return acc
    
    def predict_time(self, X_test, n_samples=100):
        """测试预测100条数据的耗时"""
        X_sample = X_test[:n_samples] if len(X_test) >= n_samples else X_test
        X_sample_tfidf = self.vectorizer.transform(X_sample)
        
        start_time = time.time()
        _ = self.model.predict(X_sample_tfidf)
        elapsed_time = time.time() - start_time
        
        return elapsed_time
    
    def save_model(self, path=None):
        """保存模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "svm_model.pkl")
        joblib.dump({
            'model': self.model,
            'vectorizer': self.vectorizer
        }, path)
        print(f"模型已保存至: {path}")
    
    def load_model(self, path=None):
        """加载模型"""
        if path is None:
            path = os.path.join(MODEL_DIR, "svm_model.pkl")
        data = joblib.load(path)
        self.model = data['model']
        self.vectorizer = data['vectorizer']
        print(f"模型已从 {path} 加载")


if __name__ == "__main__":
    from data_loader import load_data, data_analysis, split_data, prepare_data_for_svm
    
    # 加载和处理数据
    df = load_data()
    data_analysis(df)
    train_df, valid_df, test_df = split_data(df)
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data_for_svm(
        train_df, valid_df, test_df
    )
    
    # 训练和评估
    classifier = SVMClassifier()
    classifier.train(X_train, y_train, X_valid, y_valid)
    acc = classifier.evaluate(X_test, y_test)
    pred_time = classifier.predict_time(X_test)
    print(f"预测100条耗时: {pred_time:.4f}秒")
    classifier.save_model()

