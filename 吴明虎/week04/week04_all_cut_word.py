# 作业
# #目标输出;顺序不重要
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


def backtrack_cut(sentence, s_len, w_len , Dict, line ,start,global_result):
    '''
    使用回溯算法，每次获得一个词和字典比对，如果在字典内部就加入到临时line，并传入下一层，通过start游标保证不重复

    '''
    res_sent_len = s_len - start
    if res_sent_len < 1:
        global_result.append(line[:])  # line[:]做一个copy而不是使用引用作为append对象
        return global_result
    i=0
    for j in range(w_len):
        if start +j +1 > s_len:
            break
        sub_sentence = sentence[start : start +j +1]
        if sub_sentence in Dict:   #不在词表不予计算
            line.append(sub_sentence)
            global_result = backtrack_cut(sentence,s_len,w_len,Dict,line,start+j+1,global_result)
            line.pop()
    return global_result





def all_cut(sentence, Dict,global_result):
    '''
    实现全切分函数，输出根据字典能够切分出的所有的切分方式
    一个词会出现选择和不选的状态，使用动态规划
    通过判断某个词在不在字典进行剪切
    '''
    line = []
    word_len = 1
    s_len = len(sentence)
    #切的词的最大长度是 最长key的长度
    for key in Dict.keys():
        word_len = max(word_len, len(key))
    result = backtrack_cut(sentence,s_len,word_len,Dict,line,0,global_result)
    return result


def compare_nested_lists(list1, list2):
    """
    比较两个嵌套列表的内容是否一致（忽略顺序）
    """
    # 如果长度不同，直接返回False
    if len(list1) != len(list2):
        return '列表长度与目标【不】同'

    # 对两个列表进行排序
    sorted_list1 = sorted(list1)
    sorted_list2 = sorted(list2)

    # 逐元素比较
    for elem1, elem2 in zip(sorted_list1, sorted_list2):
        if elem1 != elem2:
            return '与答案内容【不】一致'
    return '与答案内容一致'

if __name__ == '__main__':
    global_result = []
    result=all_cut(sentence, Dict,global_result)
    print(result)
    print('输出结果，',compare_nested_lists(result,target))
