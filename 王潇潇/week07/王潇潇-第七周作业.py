import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)  # 忽略无预测样本警告

# 1. 加载原始数据集
df = pd.read_csv("文本分类练习.csv")
# 初始化TF-IDF向量器
tfidf = TfidfVectorizer(
    max_features=2000, 
    ngram_range=(1, 2), 
    lowercase=False  
)
# 直接用原始“review”列生成特征矩阵
X = tfidf.fit_transform(df["review"]).toarray()
y = df["label"].values  

# 划分训练集（70%）和测试集（30%），保证标签分布一致
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\n特征提取结果：")
print(f"训练集规模：{X_train.shape[0]} 条样本，{X_train.shape[1]} 个特征")
print(f"测试集规模：{X_test.shape[0]} 条样本，{X_test.shape[1]} 个特征")
def train_eval_model(model, model_name):
    """训练模型并返回评估指标"""
    # 训练模型
    model.fit(X_train, y_train)
    # 测试集预测
    y_pred = model.predict(X_test)
    # 计算4个核心指标
    metrics = {
        "模型名称": model_name,
        "准确率(Accuracy)": round(accuracy_score(y_test, y_pred), 4),
        "精确率(Precision)": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "召回率(Recall)": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1值": round(f1_score(y_test, y_pred, zero_division=0), 4)
    }
    return metrics, model

# 初始化3种经典分类模型
models = [
    (LogisticRegression(max_iter=1000, random_state=42), "逻辑回归"),
    (MultinomialNB(), "朴素贝叶斯"),
    (SVC(kernel="linear", probability=True, random_state=42), "线性SVM")
]

# 训练所有模型并收集结果
results = []
best_model = None
best_f1 = 0 

for model, name in models:
    print(f"\n正在训练 {name}...")
    metric, trained_model = train_eval_model(model, name)
    results.append(metric)
    # 更新最优模型
    if metric["F1值"] > best_f1:
        best_f1 = metric["F1值"]
        best_model = trained_model

# 转换结果为DataFrame
result_df = pd.DataFrame(results)
print("\n" + "="*80)
print("无文本预处理：各模型性能对比表")
print("="*80)
print(result_df.to_string(index=False))
print(f"\n最优模型：{result_df.loc[result_df['F1值'].idxmax(), '模型名称']}（F1值：{best_f1}）")
def predict_raw_review(review_text, model, tfidf):
    """直接用原始评论预测，无需预处理"""
    # 原始文本→TF-IDF特征（复用训练好的向量器）
    X_pred = tfidf.transform([review_text]).toarray()
    # 预测标签和置信度
    label = model.predict(X_pred)[0]
    prob = model.predict_proba(X_pred)[0][label]  # 置信度（0~1）
    return {
        "原始输入评论": review_text,
        "预测结果": "好评" if label == 1 else "差评",
        "置信度": round(prob, 4)
    }

# 测试3条原始评论
test_raw_reviews = [
    "送餐超级快！汤还是热的，味道比店里吃还香～",
    "等了1个半小时，饭全凉了！菜还少了一份，客服也不回复！",
    "包装很严实，没有撒漏，分量足够，下次还点！"
]

print("最优模型：原始文本直接预测示例")
predict_results = [predict_raw_review(txt, best_model, tfidf) for txt in test_raw_reviews]
predict_df = pd.DataFrame(predict_results)
print(predict_df.to_string(index=False))
