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
    # 内部定义回溯函数
    def backtrack(start, path):
        # 递归终止条件,语句相等
        if start == len(sentence):
            target.append(path[:])
            return
        # 尝试所有可能的切分,range 左闭右开【1-8），第一次start=0,end范围[1, 2, 3, 4, 5, 6, 7],
        # word范围[0,7]，Python切片是左闭右开
        # end = 1: word = sentence[0:1] = "经"
        # end = 2: word = sentence[0:2] = "经常"
        # end = 3: word = sentence[0:3] = "经常有"(不在词典)
        # end = 4: word = sentence[0:4] = "经常有意"(不在词典)
        # end = 5: word = sentence[0:5] = "经常有意见"(不在词典)
        # end = 6: word = sentence[0:6] = "经常有意见分"(不在词典)
        # end = 7: word = sentence[0:7] = "经常有意见分歧"(不在词典)
        for end in range(start + 1, len(sentence) + 1):
            word = sentence[start:end]
            # 如果当前词在词典中，继续递归
            if word in Dict:
                path.append(word)  # 选择
                backtrack(end, path)  # 递归
                path.pop()  # 撤销选择

    # 初始化结果列表并开始回溯
    target = []
    backtrack(0, [])
    return target

#目标输出;顺序不重要
target1 = [
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

target2=all_cut(sentence, Dict)
print(len(target1))
print(len(target2))

