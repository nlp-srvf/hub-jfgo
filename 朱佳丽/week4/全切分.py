#week4作业

#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}

#待切分文本
sentence = "经常有意见分歧"

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    """
    使用带备忘录的递归实现全切分。
    :param sentence: 待切分的字符串
    :param Dict: 词典
    :return: 一个包含所有切分方式的列表，每种方式是一个词列表
    """
    # memo用于缓存已经计算过的子问题的结果，避免重复计算，提高效率
    memo = {}

    def _recursive_cut(s):
        # 如果s的结果已经计算过，直接从缓存返回
        if s in memo:
            return memo[s]

        # 基本情况：如果字符串为空，表示一个切分路径已经完成。
        # 返回[[]]是为了方便上层调用进行拼接。
        # 例如，当 s="分歧", prefix="分歧" 时, _recursive_cut("")会返回[[]]
        # 这样 [prefix] + sub_path 就会变成 ["分歧"] + [] = ["分歧"]
        if not s:
            return [[]]

        # 存放当前字符串s的所有切分结果
        result = []
        
        # 尝试所有可能的前缀
        # range(1, len(s) + 1) 保证能取到从s[0]到s[0:len(s)]的所有前缀
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            
            # 如果前缀在词典中
            if prefix in Dict:
                # 获取剩余部分的字符串
                suffix = s[i:]
                
                # 递归地对剩余部分进行全切分
                suffix_cuts = _recursive_cut(suffix)
                
                # 将当前前缀与剩余部分的所有切分结果组合
                for sub_path in suffix_cuts:
                    result.append([prefix] + sub_path)
        
        # 将当前字符串s的计算结果存入缓存
        memo[s] = result
        return result

    # 从完整的句子开始递归
    return _recursive_cut(sentence)

# --- 执行并打印结果 ---
target = all_cut(sentence, Dict)

# 为了方便比对，将结果排序后打印
# 先按列表长度排序，再按列表内容排序
sorted_target = sorted(target, key=lambda x: (len(x), x))
for item in sorted_target:
    print(item)



#目标输出;顺序不重要
target = [
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
