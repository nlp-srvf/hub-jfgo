# coding: utf-8

"""
FastText：定义神经网络模型结构
"""

import os
import tempfile
import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

# FastText 可选支持
FASTTEXT_AVAILABLE = False
try:
    from fasttext import train_supervised
    FASTTEXT_AVAILABLE = True
except ImportError:
    warnings.warn("FastText 未安装。如需使用 FastText，请运行: pip install fasttext")

class FastTextClassifier:
    def __init__(self, config):
        self.config = config
        self.model = None
        # 如果使用 LogisticRegression 作为替代，设置相关参数
        if not FASTTEXT_AVAILABLE:
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95,
                analyzer='char_wb'
            )
            self.lr_model = LogisticRegression(
                max_iter=1000,
                random_state=config.random_seed
            )
            self.use_fasttext = False
        else:
            self.use_fasttext = True


    def preprocess_text(self, texts):
        """文本预处理"""
        processed_texts = []
        for text in texts:
            # 简单的文本清洗，去除特殊字符但保留中文和空格
            import re
            text = re.sub(r'[^\u4e00-\u9fa5\s]', '', str(text))
            processed_texts.append(' '.join(text.split()))
        return processed_texts

    def train(self, X_train, y_train):
        """训练FastText模型"""
        start_time = time.time()
        train_start_time = time.time()
        # 创建临时文件
        train_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')
        self.prepare_data(X_train, y_train, train_file.name)
        train_file.close()
        # 训练模型
        self.model = train_supervised(
            input=train_file.name,
            lr=self.config.fasttext_lr,
            epoch=self.config.fasttext_epochs,
            wordNgrams=self.config.fasttext_word_ngrams,
            verbose=2
        )
        # 删除临时文件
        os.unlink(train_file.name)
        training_time = time.time() - train_start_time
        print(f'FastText training completed in {training_time:.2f} seconds')
        self.training_time = time.time() - start_time
        return training_time

    def predict(self, texts):
        """预测"""
        start_time = time.time()
        predictions = []
        probabilities = []
        for text in texts:
            # FastText返回标签和概率
            labels, probs = self.model.predict(text)
            # 从 __label__0 或 __label__1 提取标签
            pred_label = 1 if labels[0] == '__label__1' else 0
            predictions.append(pred_label)
            # FastText返回的概率是列表，我们取对应类别的概率
            if pred_label == 1:
                probabilities.append(probs[0])
            else:
                probabilities.append(1 - probs[0])
        inference_time = time.time() - start_time
        return np.array(predictions), np.array(probabilities), inference_time

    def evaluate(self, X_test, y_test):
        """评估模型"""
        predictions, probabilities, inference_time = self.predict(X_test)
        # 计算指标
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions, output_dict=True)
        return accuracy, report, inference_time

    def save_model(self, path):
        """保存模型"""
        if self.model:
            self.model.save_model(path)
            print(f'FastText model saved to {path}')
        else:
            print('No model to save. Train the model first.')

    def load_model(self, path):
        """加载模型"""
        from fasttext import load_model
        self.model = load_model(path)
        print(f'FastText model loaded from {path}')