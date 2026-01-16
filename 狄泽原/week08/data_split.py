# -*- coding: utf-8 -*-
import json
import random
from collections import defaultdict




# ===================== 核心函数 =====================
def load_raw_data(raw_path):
    """加载原始数据，按标签分组"""
    label2questions = defaultdict(list)
    with open(raw_path, encoding="utf8") as f:
        for line in f:
            line = json.loads(line.strip())
            label = line["target"]
            questions = line["questions"]
            # 过滤空问句，去重
            questions = [q.strip() for q in questions if q.strip()]
            if len(questions) >= MIN_QUESTIONS_PER_LABEL:
                label2questions[label].extend(questions)
            else:
                print(f"警告：标签[{label}]的有效问句数不足{MIN_QUESTIONS_PER_LABEL}，跳过")

    # 对每个标签的问句去重
    for label in label2questions:
        label2questions[label] = list(set(label2questions[label]))
    return label2questions


def split_train_test(label2questions, test_ratio=0.2, seed=42):
    """拆分训练/测试集（按标签内问句拆分，避免数据泄露）"""
    random.seed(seed)
    train_data = defaultdict(list)
    test_questions = defaultdict(list)  # 每个标签的测试问句

    for label, questions in label2questions.items():
        # 打乱问句
        random.shuffle(questions)
        # 计算测试集数量（至少保留2个给训练集）
        test_size = max(1, int(len(questions) * test_ratio))
        train_size = len(questions) - test_size
        if train_size < MIN_QUESTIONS_PER_LABEL:
            # 训练集不足，减少测试集数量
            test_size = len(questions) - MIN_QUESTIONS_PER_LABEL
            train_size = MIN_QUESTIONS_PER_LABEL

        # 拆分
        train_data[label] = questions[:train_size]
        test_questions[label] = questions[train_size:train_size + test_size]

    return train_data, test_questions


def generate_test_triplets(test_questions, num_per_label=5, seed=42):
    """为测试集生成静态三元组（Anchor, Positive, Negative）"""
    random.seed(seed)
    test_triplets = []
    all_labels = list(test_questions.keys())
    if len(all_labels) < 2:
        raise ValueError("测试集标签数必须≥2（需正/负标签）")

    for pos_label in all_labels:
        pos_qs = test_questions[pos_label]
        if len(pos_qs) < 2:
            print(f"警告：测试集标签[{pos_label}]的问句数不足2，跳过生成三元组")
            continue

        # 为当前标签生成指定数量的三元组
        for _ in range(num_per_label):
            # 选Anchor和Positive（同一标签）
            anchor, positive = random.sample(pos_qs, 2)
            # 选Negative（不同标签）
            neg_labels = [l for l in all_labels if l != pos_label]
            neg_label = random.choice(neg_labels)
            neg_qs = test_questions[neg_label]
            if len(neg_qs) == 0:
                continue
            negative = random.choice(neg_qs)
            # 构造三元组（格式：[锚样本, 正样本, 负样本, 正标签]）
            test_triplets.append([anchor, positive, negative, pos_label])

    return test_triplets


def save_data(train_data, test_triplets, train_path, test_path):
    """保存训练集和测试集"""
    # 保存训练集（原始格式：{"questions": [...], "target": "..."}）
    with open(train_path, "w", encoding="utf8") as f:
        for label, questions in train_data.items():
            line = json.dumps({"questions": questions, "target": label}, ensure_ascii=False)
            f.write(line + "\n")

    # 保存测试集（三元组格式：[锚样本, 正样本, 负样本, 正标签]）
    with open(test_path, "w", encoding="utf8") as f:
        for triplet in test_triplets:
            line = json.dumps(triplet, ensure_ascii=False)
            f.write(line + "\n")

    print(f"训练集已保存至：{train_path}（共{len(train_data)}个标签）")
    print(f"测试集已保存至：{test_path}（共{len(test_triplets)}个三元组）")


# ===================== 主流程 =====================
if __name__ == "__main__":

    # ===================== 配置参数 =====================
    RAW_DATA_PATH = "../data/data.json"  # 你的原始数据文件路径
    TRAIN_OUTPUT_PATH = "train_triplet.json"  # 训练集输出路径
    TEST_OUTPUT_PATH = "test_triplet.json"  # 测试集输出路径
    TEST_RATIO = 0.2  # 测试集占比（如20%）
    RANDOM_SEED = 42  # 随机种子（保证划分结果可复现）
    MIN_QUESTIONS_PER_LABEL = 2  # 每个标签至少保留的问句数（需≥2，否则无法生成三元组）
    TRIPLETS_PER_TEST_LABEL = 5  # 每个标签在测试集中生成的三元组数

    # 1. 加载原始数据
    print("加载原始数据...")
    label2questions = load_raw_data(RAW_DATA_PATH)
    if not label2questions:
        raise ValueError("原始数据加载失败，无有效标签")

    # 2. 拆分训练/测试集
    print("拆分训练/测试集...")
    train_data, test_questions = split_train_test(
        label2questions,
        test_ratio=TEST_RATIO,
        seed=RANDOM_SEED
    )

    # 3. 生成测试集三元组
    print("生成测试集三元组...")
    test_triplets = generate_test_triplets(
        test_questions,
        num_per_label=TRIPLETS_PER_TEST_LABEL,
        seed=RANDOM_SEED
    )

    # 4. 保存文件
    save_data(train_data, test_triplets, TRAIN_OUTPUT_PATH, TEST_OUTPUT_PATH)

    # 打印统计信息
    print("\n===== 数据划分统计 =====")
    print(f"总标签数：{len(label2questions)}")
    print(f"训练集标签数：{len(train_data)}")
    print(f"测试集标签数：{len(test_questions)}")
    print(f"测试集三元组数：{len(test_triplets)}")

    # 示例输出（验证格式）
    print("\n===== 示例数据 =====")
    print("训练集示例：", json.dumps(list(train_data.items())[0], ensure_ascii=False))
    print("测试集三元组示例：", json.dumps(test_triplets[0], ensure_ascii=False))