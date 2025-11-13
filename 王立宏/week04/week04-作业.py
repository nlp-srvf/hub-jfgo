# coding:utf8

'''
实现句子全切分
'''


def all_cut(sentence, Dict):
    '''
    实现全切分函数，输出根据字典能够切分出的所有的切分方式
    :param sentence: 待切分的句子
    :param Dict: 词典
    :return: 切分后的数据
    '''
    # 词典转换为词列表
    vocab = []
    for key in Dict:
        vocab.append(key)
    # 获取待切分句子的长度，用于判断是否完成切分
    n = len(sentence)
    # 最终结果
    target = []
    def backtrack(start, path):
        if start == n: # 句子已经切分完成
            target.append(path[:])
            return
        for end in range(start + 1, n + 1):
            word = sentence[start:end]
            if word in vocab:
                # 继续向后切分
                backtrack(end, path + [word])
    backtrack(0, [])
    return target


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

if __name__ == '__main__':
    # 词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
    Dict = {
        "经常": 0.1,
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
        "分": 0.1
    }

    # 待切分文本
    sentence = "经常有意见分歧"
    result = all_cut(sentence, Dict)
    for one in result:
        print(one)
