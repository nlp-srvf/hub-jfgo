#week04 作业，实现句子全切分

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

# 采用回溯算法
def all_cut1(sentence,Dict):

    def backtrack(start,path,result):
        if start==len(sentence):
            result.append(path[:])
            return
        for end in range(start+1,len(sentence)+1):
            word = sentence[start:end]
            if word in Dict.keys():
                path.append(word)
                backtrack(end,path,result)
                path.pop()

    result=[]
    backtrack(0,[],result)
    return result




result=all_cut1(sentence,Dict)
print(result)
print(len(result))



