# week4作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常": 0.1,
        "经": 0.05,
        "有": 0.1,
        "常": 0.001,
        "有意见": 0.1,
        "歧": 0.001,
        "意见": 0.2,
        "分歧": 0.2,
        "见": 0.05,
        "意": 0.05,
        "见分歧": 0.05,
        "分": 0.1}

# 待切分文本
sentence = "经常有意见分歧"

# 目标输出;顺序不重要
target_example = [
    ['经常', '有意见', '分歧'],
    ['经常', '有意见', '分', '歧'],
    ['经常', '有', '意见', '分歧'],
    ['经常', '有', '意见', '分', '歧'],
    ['经常', '有', '意', '见分歧'],
    ['经常', '有', '意', '见', '分歧'],
    ['经常', '有', '意', '见', '分', '歧'],
    ['经', '常', '有意见', '分歧'],
    ['经', '常', '有意见', '分', '歧'],
    ['经', '常', '有', '意见', '分歧'],
    ['经', '常', '有', '意见', '分', '歧'],
    ['经', '常', '有', '意', '见分歧'],
    ['经', '常', '有', '意', '见', '分歧'],
    ['经', '常', '有', '意', '见', '分', '歧']
]


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    """
    使用记忆化搜索（动态规划）实现全切分
    """

    # memo 字典用于缓存，避免重复计算
    # memo[i] 存储的是 sentence[i:] (从索引i到末尾的子字符串) 的所有切分方式
    memo = {}
    n = len(sentence)

    def dfs(start_index):
        """
        递归函数，返回 sentence[start_index:] 的所有切分路径
        """

        # 1. 检查缓存
        # 如果这个索引的结果已经计算过，直接返回，避免重复劳动
        if start_index in memo:
            return memo[start_index]

        # 2. 基本情况（递归终止条件）
        # 如果起始索引已经到达句子末尾，说明找到了一条完整的切分路径
        # 返回 [[]]：一个包含“空路径”的列表。
        # 这使得上层调用可以方便地在 "空路径" 前面添加单词。
        if start_index == n:
            return [[]]

        # 3. 递归步骤
        # 存储从 start_index 开始的所有切分路径
        all_paths = []

        # 尝试从 start_index 往后的所有可能的词
        # end_index 是结束位置（不包含）
        for end_index in range(start_index + 1, n + 1):
            # 截取候选词
            word = sentence[start_index:end_index]

            # 检查这个词是否在词典中
            if word in Dict:
                # 如果在词典中，这是一个有效的词
                # 我们需要递归地找到剩余部分 (sentence[end_index:]) 的所有切分方式
                sub_paths = dfs(end_index)

                # 遍历剩余部分的所有切分方式
                for path in sub_paths:
                    # 将当前词 [word] 和剩余部分的切分 [path] 组合起来
                    all_paths.append([word] + path)

        # 4. 缓存结果并返回
        # 将从 start_index 开始的所有路径存入缓存
        memo[start_index] = all_paths
        return all_paths

    # 从句子的第0个位置开始递归
    target = dfs(0)
    return target


# --- 执行代码并验证 ---
result = all_cut(sentence, Dict)

print("--- 计算结果 ---")
for r in sorted(result):  # 排序以便于查看
    print(r)

print(f"\n总共 {len(result)} 种切分方式。")

# 验证是否和目标一致 (顺序可能不同)
# 将列表转换为元组的集合，以便不关心顺序地进行比较
result_set = set(tuple(r) for r in result)
target_set = set(tuple(t) for t in target_example)

if result_set == target_set:
    print("验证成功：计算结果与目标输出完全一致！")
else:
    print("验证失败：计算结果与目标输出不一致。")
    print("目标中缺失的结果：", target_set - result_set)
    print("计算多出的结果：", result_set - target_set)
