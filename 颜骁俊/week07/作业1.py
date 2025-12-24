import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 尝试导入中文分词库 jieba
try:
    import jieba

    HAS_JIEBA = True
except ImportError:
    HAS_JIEBA = False
    print("未检测到 jieba 库，将使用字级别分词。建议安装: pip install jieba")


def load_and_process_data(file_path):
    """
    加载并清洗数据
    """
    try:
        # 尝试默认编码读取
        df = pd.read_csv(file_path)
    except UnicodeDecodeError:
        # 中文CSV常遇到的编码问题，尝试gbk
        df = pd.read_csv(file_path, encoding='gbk')
    except Exception as e:
        print(f"读取文件失败: {e}")
        return None

    # 检查必要的列
    if 'review' not in df.columns or 'label' not in df.columns:
        print("错误：数据集中缺少 'review' 或 'label' 列")
        return None

    # 删除空值
    df = df.dropna(subset=['review', 'label'])

    # 确保文本列是字符串格式
    df['review'] = df['review'].astype(str)

    return df


def get_tokenizer():
    """
    获取分词器：如果有jieba则用jieba，否则用字级别
    """
    if HAS_JIEBA:
        def tokenizer(text):
            return list(jieba.cut(text))

        return tokenizer
    else:
        def tokenizer(text):
            return list(text)  # 将每个字符作为一个token

        return tokenizer


# --- 主程序 ---
if __name__ == "__main__":
    file_path = '文本分类练习.csv'  # 请确保文件在当前目录下

    print("1. 正在加载数据...")
    df = load_and_process_data(file_path)

    if df is not None:
        print(f"   数据加载成功，共 {len(df)} 条记录。")
        print(f"   标签分布:\n{df['label'].value_counts()}")

        # 2. 划分训练集和测试集
        print("2. 正在划分训练集与测试集...")
        X = df['review']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. 特征工程 (TF-IDF)
        print("3. 正在进行文本向量化 (TF-IDF)...")
        tokenizer = get_tokenizer()
        # token_pattern=None 是为了避免sklearn默认正则过滤掉单字，因为我们已经自定义了tokenizer
        vectorizer = TfidfVectorizer(tokenizer=tokenizer, token_pattern=None, max_features=5000)

        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        # 4. 模型训练 (逻辑回归)
        print("4. 正在训练逻辑回归模型...")
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train_tfidf, y_train)

        # 5. 预测与评估
        print("5. 正在评估模型...")
        y_pred = clf.predict(X_test_tfidf)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print("\n" + "=" * 30)
        print(f"实验结果 (Accuracy): {acc:.4f}")
        print("=" * 30)
        print("\n详细分类报告:")
        print(report)

        # 简单的单条测试
        print("\n测试样例:")
        test_samples = ["这家的菜太难吃了，送餐也慢", "味道非常棒，下次还来", "一般般，无功无过"]
        sample_vec = vectorizer.transform(test_samples)
        sample_preds = clf.predict(sample_vec)
        for text, pred in zip(test_samples, sample_preds):
            label_name = "正面 (1)" if pred == 1 else "负面 (0)"
            print(f"文本: {text} -> 预测: {label_name}")
