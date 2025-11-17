# week3作业

# 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改


# 实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    words = list(Dict.keys())
    print('words: ', words)
    results = []

    def dg(start, path):
        if start == len(sentence):
            results.append(path)
            return

        for end in range(start, len(sentence)):
            end += 1
            word = sentence[start:end]
            # print(word)
            if word in words:
                dg(end, path + [word])

    dg(0, [])
    return results


"""
# 目标输出;顺序不重要
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
"""

if __name__ == '__main__':
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

    results = all_cut(sentence, Dict)
    for i, result in enumerate(results):
        print(f'{++i}, --> {result}')
