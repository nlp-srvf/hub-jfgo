# coding: utf-8

'''
evaluate.py：模型评估函数，支持多次重复评估和k折交叉验证
'''

import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json

def compute_embeddings(model, data_loader, device):
    """
    计算数据集中所有文本的嵌入表示
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
    Returns:
        embeddings: 所有文本的嵌入表示
        labels: 对应的标签
        targets: 对应的目标名称
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            texts = batch['texts'].to(device)
            labels = batch['labels'].to(device)
            targets = batch['targets']
            
            # 获取文本嵌入
            embeddings = model.encode(texts)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_targets.extend(targets)
    
    all_embeddings = np.vstack(all_embeddings)
    all_labels = np.hstack(all_labels)
    
    return all_embeddings, all_labels, all_targets

def predict_with_similarity(query_embedding, candidate_embeddings, threshold=0.5):
    """
    基于相似度进行预测
    Args:
        query_embedding: 查询文本嵌入
        candidate_embeddings: 候选文本嵌入
        threshold: 相似度阈值
    Returns:
        predicted_labels: 预测的标签
        similarities: 相似度分数
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # 计算余弦相似度
    similarities = cosine_similarity(
        query_embedding.reshape(1, -1),
        candidate_embeddings
    )[0]
    
    # 返回最相似的样本标签（实际应用中需要更复杂的策略）
    best_match_idx = np.argmax(similarities)
    confidence = similarities[best_match_idx]
    
    return best_match_idx, confidence, similarities

def evaluate_model(model, data_loader, device, threshold=0.5):
    """
    评估模型性能
    Args:
        model: 训练好的模型
        data_loader: 验证数据加载器
        device: 计算设备
        threshold: 相似度阈值
    Returns:
        metrics: 评估指标字典
    """
    model.eval()
    
    # 计算所有嵌入
    embeddings, labels, targets = compute_embeddings(model, data_loader, device)
    
    # 基于嵌入进行分类（这里简化为用最近邻）
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score
    
    # 使用KNN进行分类
    knn = KNeighborsClassifier(n_neighbors=3, metric='cosine')
    
    # 交叉验证评估
    cv_scores = cross_val_score(knn, embeddings, labels, cv=5)
    
    # 训练KNN并预测
    knn.fit(embeddings, labels)
    predicted_labels = knn.predict(embeddings)
    
    # 计算指标
    accuracy = accuracy_score(labels, predicted_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predicted_labels, average='weighted'
    )
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    return metrics, embeddings, labels, predicted_labels

def evaluate_by_triplet_loss(model, data_loader, device, margin=1.0):
    """
    基于三元组损失评估模型性能
    Args:
        model: 训练好的模型
        data_loader: 训练数据加载器（包含三元组）
        device: 计算设备
        margin: 三元组损失的边界值
    Returns:
        avg_loss: 平均三元组损失
        accuracy: 三元组准确率
    """
    model.eval()
    total_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    criterion = torch.nn.TripletMarginLoss(margin=margin)
    
    with torch.no_grad():
        for batch in data_loader:
            anchors = batch['anchors'].to(device)
            positives = batch['positives'].to(device)
            negatives = batch['negatives'].to(device)
            
            # 获取嵌入
            anchor_emb, positive_emb, negative_emb = model(anchors, positives, negatives)
            
            # 计算三元组损失
            loss = criterion(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            
            # 计算准确率（正样本距离 < 负样本距离 - margin）
            pos_dist = torch.pairwise_distance(anchor_emb, positive_emb, p=2)
            neg_dist = torch.pairwise_distance(anchor_emb, negative_emb, p=2)
            
            correct = (pos_dist < neg_dist - margin).sum().item()
            correct_predictions += correct
            total_samples += anchors.size(0)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct_predictions / total_samples
    
    return avg_loss, accuracy

def compute_class_embeddings(model, class_examples, vocab, config, device):
    """
    计算每个类别的平均嵌入表示
    Args:
        model: 训练好的模型
        class_examples: 每个类别的示例文本字典
        vocab: 词汇表
        config: 配置字典
        device: 计算设备
    Returns:
        class_embeddings: 类别嵌入表示字典
    """
    model.eval()
    class_embeddings = {}
    
    with torch.no_grad():
        for class_name, examples in class_examples.items():
            embeddings = []
            
            for text in examples:
                # 转换文本为索引
                indices = vocab.text_to_indices(text, config['max_length'])
                input_tensor = torch.tensor([indices], dtype=torch.long).to(device)
                
                # 获取嵌入
                embedding = model.encode(input_tensor)
                embeddings.append(embedding.cpu().numpy())
            
            # 计算平均嵌入
            if embeddings:
                class_embeddings[class_name] = np.mean(embeddings, axis=0)
    
    return class_embeddings

def save_evaluation_results(metrics, output_path):
    """
    保存评估结果
    Args:
        metrics: 评估指标字典
        output_path: 输出文件路径
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def print_evaluation_results(metrics, dataset_name="Validation"):
    """
    打印评估结果
    Args:
        metrics: 评估指标字典
        dataset_name: 数据集名称
    """
    print(f"\n{dataset_name} Set Results:")
    print("=" * 50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    if 'cv_mean' in metrics:
        print(f"CV Mean:   {metrics['cv_mean']:.4f} ± {metrics['cv_std']:.4f}")
    print("=" * 50)

if __name__ == "__main__":
    # 测试代码
    from config import Config
    from loader import get_data_loaders
    
    # 这里需要先训练模型才能进行评估
    # 仅供测试函数结构
    print("评估函数已定义，需要训练模型后才能使用")
