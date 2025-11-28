#week3作业
#词典；每个词后方存储的是其词频，词频仅为示例，不会用到，也可自行修改
import pprint

DICT = {"经常":0.1,
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



#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    #TODO
    dp = [[None for _ in range(len(sentence) + 1)] for _ in range(len(sentence) + 1)]
    for i in range(len(sentence)):
        dp[i][i] = {'valid': True, 'words': [[]]}
        if sentence[i:i+1] in Dict:
            dp[i][i+1] = {'valid': True, 'words': [[sentence[i:i+1]]]}
    # pprint.pprint(dp)
    for incre in range(2, len(dp)):
        for i in range(0, len(dp) - incre):
            words = []
            is_valid = False
            if sentence[i : i + incre] in Dict:
                is_valid = True
                words.append([sentence[i : i + incre]])

            for k in range(1, incre):
                if dp[i][i + k]['valid'] and dp[i + k][i + incre]['valid']:
                    is_valid = True
                    for ik in dp[i][i + k]['words']:
                        for ki in dp[i + k][i + incre]['words']:
                            if (ik+ki) not in words:
                                words.append(ik+ki)
            if is_valid:
                dp[i][i + incre] = {'valid': True, 'words': words}
                # print(i,',',(i+incre),' find ', str(words))
    if dp[0][len(sentence)]['valid']:
        return dp[0][len(sentence)]['words']
    else:
        return None

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

if __name__ == '__main__':
    #待切分文本
    sentence_to_cut = "经常有意见分歧"
    targets = all_cut(sentence_to_cut, DICT)
    pprint.pprint(targets)

