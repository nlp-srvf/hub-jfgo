# coding: utf-8

'''
evaluate.py：模型评估函数，支持多次重复评估和k折交叉验证
'''

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import roc_auc_score

def calculate_metrics(y_true, y_pred, y_prob=None):
    """
    计算评估指标
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param y_prob: 预测概率（可选）
    :return: 指标字典
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    # 计算各类别的指标
    precision_per_class = precision_score(y_true, y_pred, average=None)
    recall_per_class = recall_score(y_true, y_pred, average=None)
    f1_per_class = f1_score(y_true, y_pred, average=None)

    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'precision_0': precision_per_class[0],  # 差评
        'precision_1': precision_per_class[1],  # 好评
        'recall_0': recall_per_class[0],
        'recall_1': recall_per_class[1],
        'f1_0': f1_per_class[0],
        'f1_1': f1_per_class[1]
    }

    # 如果提供了预测概率，可以计算AUC
    if y_prob is not None:
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(y_true)) == 2:  # 二分类
                auc = roc_auc_score(y_true, y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob[:, 1])
                results['auc'] = auc
        except:
            pass  # AUC计算失败时忽略
    return results

def evaluate_model(classifier, eval_dataloader=None, X_test=None, y_test=None):
    """
    评估模型
    支持两种不同的输入方式：
    1. 深度学习模型：传入dataloader
    2. 传统机器学习模型：直接传入X_test, y_test
    """
    if eval_dataloader is not None and X_test is None and y_test is None:
        # 深度学习模型评估
        accuracy, report, inference_time = classifier.evaluate(eval_dataloader)
        # 重新计算指标
        predictions, probabilities, _ = classifier.predict(eval_dataloader)
        true_labels = []
        for batch in eval_dataloader:
            if isinstance(batch, dict):  # BERT格式
                true_labels.extend(batch['labels'].numpy())
            else:  # 其他格式
                _, labels = batch
                true_labels.extend(labels.numpy())
        metrics = calculate_metrics(true_labels, predictions)
        metrics['inference_time'] = inference_time

    elif X_test is not None and y_test is not None:
        # 传统机器学习模型评估
        accuracy, report, inference_time = classifier.evaluate(X_test, y_test)
        # 获取预测结果
        predictions, probabilities, _ = classifier.predict(X_test)
        metrics = calculate_metrics(y_test, predictions, probabilities)
        metrics['inference_time'] = inference_time
    else:
        raise ValueError("Either dataloader or (X_test, y_test) must be provided")
    return metrics, accuracy

def print_evaluation_results(model_name, results):
    """打印评估结果"""
    print("=" * 50)
    print(f"{model_name} 评估结果:")
    print("=" * 50)
    print(f"准确率 (Accuracy): {results['accuracy']:.4f}")
    print(f"精确率 (Weighted Precision): {results['precision']:.4f}")
    print(f"召回率 (Weighted Recall): {results['recall']:.4f}")
    print(f"F1值 (Weighted F1): {results['f1']:.4f}")
    print("-" * 50)
    print("各类别指标:")
    print(f"差评 (label=0) - 精确率: {results['precision_0']:.4f}, 召回率: {results['recall_0']:.4f}, F1: {results['f1_0']:.4f}")
    print(f"好评 (label=1) - 精确率: {results['precision_1']:.4f}, 召回率: {results['recall_1']:.4f}, F1: {results['f1_1']:.4f}")
    if 'auc' in results:
        print(f"AUC: {results['auc']:.4f}")

    if 'inference_time' in results:
        print(f"推理时间: {results['inference_time']:.4f} 秒")
    print("=" * 50)

def repeat_evaluation(model_class, config, X, y, model_name,
                     vocab_builder=None, n=10, data_processor=None, **kwargs):
    """
    多次重复评估函数

    Args:
        model_class: 模型类
        config: 配置对象
        X: 特征数据
        y: 标签数据
        model_name: 模型名称
        vocab_builder: 词汇表构建函数（可选）
        n: 重复次数
        data_processor: 数据处理器（可选）
        **kwargs: 额外的关键字参数

    Returns:
        包含所有评估结果的字典
    """
    print(f"\n{'='*60}")
    print(f"开始{n}次重复评估：{model_name}")
    print(f"{'='*60}")

    # 存储所有次的结果
    all_results = []

    for i in range(n):
        print(f"\n第 {i + 1}/{n} 次评估：")

        # 每次重新分割数据，确保随机性
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size,
            random_state=config.random_seed + i,  # 使用不同的随机种子
            stratify=y
        )

        # 创建分类器实例
        if vocab_builder:
            # 为深度学习模型构建词汇表
            vocab = vocab_builder(X_train)
            classifier = model_class(config, len(vocab))
        else:
            classifier = model_class(config)

        # 训练时间测量开始
        train_start_time = time.time()

        # 训练模型
        if model_name == 'BERT':
            # BERT模型
            train_dataloader = data_processor.get_bert_dataloader(X_train, y_train)
            test_dataloader = data_processor.get_bert_dataloader(X_test, y_test, shuffle=False)

            try:
                # 训练
                classifier.train(train_dataloader)
                training_time = time.time() - train_start_time
                # 评估
                metrics, _ = evaluate_model(classifier, eval_dataloader=test_dataloader)
                metrics['training_time'] = training_time
            except Exception as e:
                print(f"BERT 第{i+1}次训练失败: {str(e)}")
                continue

        elif model_name in ['TextCNN', 'LSTM', 'GatedCNN']:
            # 其他深度学习模型
            train_dataloader = data_processor.get_traditional_dataloader(X_train, y_train, vocab)
            test_dataloader = data_processor.get_traditional_dataloader(X_test, y_test, vocab, shuffle=False)

            try:
                # 训练
                classifier.train(train_dataloader)
                training_time = time.time() - train_start_time
                # 评估
                metrics, _ = evaluate_model(classifier, eval_dataloader=test_dataloader)
                metrics['training_time'] = training_time
            except Exception as e:
                print(f"{model_name} 第{i+1}次训练失败: {str(e)}")
                continue

        else:
            # 传统机器学习模型
            try:
                classifier.train(X_train, y_train)
                training_time = time.time() - train_start_time
                # 评估
                metrics, _ = evaluate_model(classifier, X_test=X_test, y_test=y_test)
                metrics['training_time'] = training_time
            except Exception as e:
                print(f"{model_name} 第{i+1}次训练失败: {str(e)}")
                continue

        # 添加评估次数信息
        metrics['evaluation_id'] = i + 1
        metrics['model'] = model_name
        metrics['random_seed'] = config.random_seed + i

        # 保存本次评估的训练集和测试集大小
        metrics['train_size'] = len(X_train)
        metrics['test_size'] = len(X_test)

        # 重分割后正负样本数量
        pos_train = np.sum(y_train == 1)
        neg_train = np.sum(y_train == 0)
        pos_test = np.sum(y_test == 1)
        neg_test = np.sum(y_test == 0)
        metrics['pos_train'] = pos_train
        metrics['neg_train'] = neg_train
        metrics['pos_test'] = pos_test
        metrics['neg_test'] = neg_test

        all_results.append(metrics)

        # 打印单次结果（可选）
        print(f"第{i+1}次测试指标：")
        print_evaluation_results(model_name, metrics)

    # 计算统计信息
    if all_results:
        score_keys = ['accuracy', 'precision', 'recall', 'f1',
                     'precision_0', 'precision_1', 'recall_0', 'recall_1',
                     'f1_0', 'f1_1', 'training_time', 'inference_time']

        stats = {
            'evaluation_count': len(all_results),
            'model': model_name
        }

        for key in score_keys:
            if key in all_results[0]:
                values = [result[key] for result in all_results]
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)

        print(f"\n{'='*60}")
        print(f"{n}次重复评估统计结果：")
        print(f"{'='*60}")
        print(f"评估次数：{n}")
        print(f"准确率均值：{stats['accuracy_mean']:.4f} ± {stats['accuracy_std']:.4f}")
        print(f"准确率范围：[{stats['accuracy_min']:.4f}, {stats['accuracy_max']:.4f}]")
        print(f"F1值均值：{stats['f1_mean']:.4f} ± {stats['f1_std']:.4f}")
        print(f"F1值范围：[{stats['f1_min']:.4f}, {stats['f1_max']:.4f}]")

    return {
        'all_results': all_results,
        'statistics': stats if all_results else None
    }