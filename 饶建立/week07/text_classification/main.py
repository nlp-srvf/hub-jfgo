# -*- coding: utf-8 -*-
"""
主程序：运行所有模型并对比结果
"""
import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from tabulate import tabulate

from config import BATCH_SIZE, RNN_CONFIG, LSTM_CONFIG, BERT_CONFIG
from data_loader import (
    load_data, data_analysis, split_data, build_vocab,
    get_dataloaders, prepare_data_for_svm
)
from svm_model import SVMClassifier
from rnn_model import RNNTrainer
from lstm_model import LSTMTrainer
from bert_model import BertTrainer


def run_svm(X_train, y_train, X_valid, y_valid, X_test, y_test):
    """运行SVM模型"""
    print("\n" + "=" * 60)
    print("【SVM模型】")
    print("=" * 60)
    
    classifier = SVMClassifier()
    classifier.train(X_train, y_train, X_valid, y_valid)
    acc = classifier.evaluate(X_test, y_test)
    pred_time = classifier.predict_time(X_test)
    classifier.save_model()
    
    return {
        'Model': 'SVM',
        'Learning_Rate': 'N/A',
        'acc': f'{acc:.4f}',
        'time(100条)': f'{pred_time:.4f}s'
    }


def run_rnn(train_loader, valid_loader, test_loader, vocab_size):
    """运行RNN模型"""
    print("\n" + "=" * 60)
    print("【RNN模型】")
    print("=" * 60)
    
    trainer = RNNTrainer()
    trainer.train(train_loader, valid_loader, vocab_size)
    acc = trainer.evaluate(test_loader)
    pred_time = trainer.predict_time(test_loader)
    
    return {
        'Model': 'RNN',
        'Learning_Rate': str(RNN_CONFIG['learning_rate']),
        'acc': f'{acc:.4f}',
        'time(100条)': f'{pred_time:.4f}s'
    }


def run_lstm(train_loader, valid_loader, test_loader, vocab_size):
    """运行LSTM模型"""
    print("\n" + "=" * 60)
    print("【LSTM模型】")
    print("=" * 60)
    
    trainer = LSTMTrainer()
    trainer.train(train_loader, valid_loader, vocab_size)
    acc = trainer.evaluate(test_loader)
    pred_time = trainer.predict_time(test_loader)
    
    return {
        'Model': 'LSTM',
        'Learning_Rate': str(LSTM_CONFIG['learning_rate']),
        'acc': f'{acc:.4f}',
        'time(100条)': f'{pred_time:.4f}s'
    }


def run_bert(train_df, valid_df, test_df):
    """运行BERT模型"""
    print("\n" + "=" * 60)
    print("【BERT模型】")
    print("=" * 60)
    
    trainer = BertTrainer()
    trainer.build_model()
    train_loader, valid_loader, test_loader = trainer.prepare_data(train_df, valid_df, test_df)
    trainer.train(train_loader, valid_loader)
    acc = trainer.evaluate(test_loader)
    pred_time = trainer.predict_time(test_loader)
    
    return {
        'Model': 'BERT',
        'Learning_Rate': str(BERT_CONFIG['learning_rate']),
        'acc': f'{acc:.4f}',
        'time(100条)': f'{pred_time:.4f}s'
    }


def main():
    """主函数"""
    print("=" * 60)
    print("电商评论情感分类 - 模型对比实验")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n【数据加载】")
    df = load_data()
    
    # 2. 数据分析
    print("\n【数据分析】")
    stats = data_analysis(df)
    
    # 3. 数据集划分
    print("\n【数据集划分】")
    train_df, valid_df, test_df = split_data(df)
    
    # 4. 构建词表
    print("\n【构建词表】")
    vocab = build_vocab(train_df)
    vocab_size = len(vocab)
    
    # 5. 准备数据
    # SVM数据
    X_train, y_train, X_valid, y_valid, X_test, y_test = prepare_data_for_svm(
        train_df, valid_df, test_df
    )
    
    # RNN/LSTM数据
    train_loader, valid_loader, test_loader = get_dataloaders(
        train_df, valid_df, test_df, vocab, BATCH_SIZE
    )
    
    # 6. 训练和评估各模型
    results = []
    
    # SVM
    svm_result = run_svm(X_train, y_train, X_valid, y_valid, X_test, y_test)
    results.append(svm_result)
    
    # RNN
    rnn_result = run_rnn(train_loader, valid_loader, test_loader, vocab_size)
    results.append(rnn_result)
    
    # LSTM
    lstm_result = run_lstm(train_loader, valid_loader, test_loader, vocab_size)
    results.append(lstm_result)
    
    # BERT
    bert_result = run_bert(train_df, valid_df, test_df)
    results.append(bert_result)
    
    # 7. 输出对比结果
    print("\n" + "=" * 60)
    print("【模型对比结果】")
    print("=" * 60)
    
    results_df = pd.DataFrame(results)
    print("\n" + tabulate(results_df, headers='keys', tablefmt='grid', showindex=False))
    
    # 保存结果到CSV
    results_df.to_csv(os.path.join(os.path.dirname(__file__), 'results.csv'), index=False, encoding='utf-8-sig')
    print("\n结果已保存至 results.csv")
    
    return results_df


if __name__ == "__main__":
    main()

