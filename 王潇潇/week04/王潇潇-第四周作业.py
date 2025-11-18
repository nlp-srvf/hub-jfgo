#week4作业：实现全切分文本

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

# 用来输出结果的列表
li_result=list()
# 获取字典中的最大值
maxlength =0

#实现全切分函数，输出根据字典能够切分出的所有的切分方式
def all_cut(sentence, Dict):
    global maxlength
    for key in Dict.keys():
        maxlength=max(maxlength,len(key))
    # 查找集合中是否包含该元素
    # 使用回溯进行计算
    list_one = list()
    dp(sentence,list_one,0)


# 切割的字符 list当前的列表 star 开始字符位置
# 递归
def dp(sentence,list,star):
    global li_result
    if star>=len(sentence):
        # 插入的是copy值 因为调用同一个对象后面改变也会跟着改变
        li_result.append(list.copy())
        return
    # 循环
    for i in range(1,maxlength+1):
        # 需要截取的片段在词表中存在
        if sentence[star:star+i] not in Dict:
            continue;
        if star+i>len(sentence):
            return
        else:
            list.append(sentence[star:star+i])
            dp(sentence,list,star+i)
            list.remove(sentence[star:star+i])

# 调用函数
all_cut(sentence,Dict)

# 打印结果
for item in li_result:
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


