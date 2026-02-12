import re
import json
import pandas
import itertools
# from py2neo import Graph
from neo4j import GraphDatabase
from collections import defaultdict


'''
使用文本匹配的方式进行知识图谱的使用
'''

class GraphQA:
    def __init__(self):
        # self.graph = Graph("http://localhost:7474", auth=("neo4j", "neo4jbeijing"))
        self.graph = GraphDatabase.driver(
            "bolt://localhost:7687",
            auth=("neo4j", "neo4jbeijing")
        )
        schema_path = "kg_schema.json"
        templet_path = "question_templet.xlsx"
        self.load(schema_path, templet_path)
        print("知识图谱问答系统加载完毕！\n===============")

    #加载模板
    def load(self, schema_path, templet_path):
        self.load_kg_schema(schema_path)
        self.load_question_templet(templet_path)
        return

    #加载图谱信息，可以限制实体、关系、属性的范围，也可以获取高精度的实体
    # 实体不支持以数字开头，必须是字母开头，可以包含字母、数字或下划线，但不能单独是数字
    def load_kg_schema(self, path):
        with open(path, encoding="utf8") as f:
            schema = json.load(f)
        self.relation_set = set(schema["relations"])
        self.entity_set = set(schema["entitys"])
        self.label_set = set(schema["labels"])
        self.attribute_set = set(schema["attributes"])
        return

    #加载模板信息
    def load_question_templet(self, templet_path):
        dataframe = pandas.read_excel(templet_path)
        self.question_templet = []
        for index in range(len(dataframe)):
            question = dataframe["question"][index]
            cypher = dataframe["cypher"][index]
            cypher_check = dataframe["check"][index]
            answer = dataframe["answer"][index]
            self.question_templet.append([question, cypher, json.loads(cypher_check), answer])
        return


    #获取问题中谈到的实体，可以使用基于词表的方式，也可以使用NER模型
    def get_mention_entitys(self, sentence):
        return re.findall("|".join(self.entity_set), sentence)

    # 获取问题中谈到的关系，也可以使用各种文本分类模型
    def get_mention_relations(self, sentence):
        return re.findall("|".join(self.relation_set), sentence)

    # 获取问题中谈到的属性
    def get_mention_attributes(self, sentence):
        return re.findall("|".join(self.attribute_set), sentence)

    # 获取问题中谈到的标签
    def get_mention_labels(self, sentence):
        return re.findall("|".join(self.label_set), sentence)

    #对问题进行预处理，提取需要的信息
    def parse_sentence(self, sentence):
        entitys = self.get_mention_entitys(sentence)
        relations = self.get_mention_relations(sentence)
        labels = self.get_mention_labels(sentence)
        attributes = self.get_mention_attributes(sentence)
        return {"%ENT%":entitys,
                "%REL%":relations,
                "%LAB%":labels,
                "%ATT%":attributes}

    #将提取到的值分配到键上
    def decode_value_combination(self, value_combination, cypher_check):
        res = {}
        for index, (key, required_count) in enumerate(cypher_check.items()):
            if required_count == 1:
                res[key] = value_combination[index][0]
            else:
                for i in range(required_count):
                    key_num = key[:-1] + str(i) + "%"
                    res[key_num] = value_combination[index][i]
        return res

    #对于找到了超过模板中需求的实体数量的情况，需要进行排列组合
    #info:{"%ENT%":["周杰伦", "方文山"], “%REL%”:[“作曲”]}
    def get_combinations(self, cypher_check, info):
        slot_values = []
        for key, required_count in cypher_check.items():
            slot_values.append(itertools.combinations(info[key], required_count))
        value_combinations = itertools.product(*slot_values)
        combinations = []
        for value_combination in value_combinations:
            combinations.append(self.decode_value_combination(value_combination, cypher_check))
        return combinations

    #将带有token的模板替换成真实词
    #string:%ENT1%和%ENT2%是%REL%关系吗
    #combination: {"%ENT1%":"word1", "%ENT2%":"word2", "%REL%":"word"}
    def replace_token_in_string(self, string, combination):
        for key, value in combination.items():
            if not value:
                return None
            string = string.replace(key, value)
        return string

    #对于单条模板，根据抽取到的实体属性信息扩展，形成一个列表
    #info:{"%ENT%":["周杰伦", "方文山"], “%REL%”:[“作曲”]}
    def expand_templet(self, templet, cypher, cypher_check, info, answer):
        combinations = self.get_combinations(cypher_check, info)
        templet_cpyher_pair = []
        for combination in combinations:
            # print('templet:   ', templet, 'combination:   ', combination)
            replaced_templet = self.replace_token_in_string(templet, combination)
            # print('templet:   ', templet, '\t', combination, '\t', replaced_templet)
            replaced_cypher = self.replace_token_in_string(cypher, combination)
            # print('cypher:   ', templet, '\t', combination, '\t', replaced_cypher)
            replaced_answer = self.replace_token_in_string(answer, combination)
            # print('answer:   ', templet, '\t', combination, '\t', replaced_answer)
            templet_cpyher_pair.append([replaced_templet, replaced_cypher, replaced_answer])
            # print('--------------------------------')
        return templet_cpyher_pair


    #验证从文本种提取到的信息是否足够填充模板，如果不足够就跳过，节省运算速度
    # 如模板：  %ENT%和%ENT%是什么关系？  这句话需要两个实体才能填充，如果问题中只有一个，该模板无法匹配
    def check_cypher_info_valid(self, info, cypher_check):
        for key, required_count in cypher_check.items():
            if len(info.get(key, [])) < required_count:
                return False
        return True

    #根据提取到的实体，关系等信息，将模板展开成待匹配的问题文本
    def expand_question_and_cypher(self, info):
        templet_cypher_pair = []
        for templet, cypher, cypher_check, answer in self.question_templet:
            if self.check_cypher_info_valid(info, cypher_check):
                print('handling templet:   ', templet, 'cypher:   ', cypher, 'cypher_check:   ', cypher_check, 'info:   ', info, 'answer:   ', answer)
                templet_cypher_pair += self.expand_templet(templet, cypher, cypher_check, info, answer)
        return templet_cypher_pair

    #距离函数，文本匹配的所有方法都可以使用
    def sentence_similarity_function(self, string1, string2):
        # print("计算  %s %s"%(string1, string2))
        jaccard_distance = len(set(string1) & set(string2)) / len(set(string1) | set(string2))
        return jaccard_distance

    #通过问题匹配的方式确定匹配的cypher
    def cypher_match(self, sentence, info):
        templet_cypher_pair = self.expand_question_and_cypher(info)
        # print(templet_cypher_pair)
        result = []
        for templet, cypher, answer in templet_cypher_pair:
            score = self.sentence_similarity_function(sentence, templet)
            result.append([templet, cypher, score, answer])
        result = sorted(result, reverse=True, key=lambda x: x[2])
        return result

    #解析结果
    def parse_result(self, graph_search_result, answer, info):
        graph_search_result = graph_search_result[0]
        #关系查找返回的结果形式较为特殊，单独处理
        if "REL" in graph_search_result:
            # print(graph_search_result, type(graph_search_result["REL"]))
            graph_search_result["REL"] = list(graph_search_result["REL"])[1]
        answer = self.replace_token_in_string(answer, graph_search_result)
        return answer


    #对外提供问答接口
    def query(self, sentence):
        print("============")
        print(sentence)
        info = self.parse_sentence(sentence)    #信息抽取
        print("info:", info)
        templet_cypher_score = self.cypher_match(sentence, info)  #cypher匹配
        with self.graph.session() as session:
            for templet, cypher, score, answer in templet_cypher_score:
                print('executing cypher:   ', cypher, 'answer:   ', answer)
                graph_search_result = session.run(cypher).data()
                # 最高分命中的模板不一定在图上能找到答案, 当不能找到答案时，运行下一个搜索语句, 找到答案时停止查找后面的模板
                print('graph_search_result:   ', graph_search_result)
                if graph_search_result:
                    answer = self.parse_result(graph_search_result, answer, info)
                    if not answer:
                        continue
                    return answer        
        return None


if __name__ == "__main__":
    """
    知识图谱回答问题的最大难点在于主实体的识别，如果是采用大模型，由于缺少语义识别，很难正确识别出其中的主实体，从而导致答案错误。
    即使采用大模型进行语义识别和实体识别，对于链式关系的实体，仍然无法准确找到主实体。
    寻找实体的常规做法是：
    1. 一级匹配（高置信度）：
        尝试直接在 KG 已知实体表中匹配
        完全匹配或别名匹配 → 高置信度候选
    2. 二级匹配（低置信度 / 模糊匹配）：
        对未匹配到的文本片段，使用 LLM 或向量检索
        生成潜在实体候选，加入候选池
    3. 排序与打分：
        给每个候选实体计算 置信度分数
        结合：
            匹配类型（exact / alias / semantic）
            上下文关系（前文对话 / KG邻居）
            实体类型约束
        选出最终候选实体或多个候选实体
    以上代码中的主实体head和tail是基于出现频率来判定，也可以根据图库中关系的方向确定，或者使用大模型进行识别。
    识别出的实体以后，生成的查询语句必须和原始的问题进行匹配，以保证查询语义上的一致性。主要原因是使用嵌入式的模板生成的查询语句可能存在语义偏差，甚至不合逻辑。
    生成最可能的查询语句，通常可以对模板和语句进行全排列。

    实体名不能有任务偏差，否则会导致查询语句不准确，无法获取结果
    """
    graph = GraphQA()
    # res = graph.query("珊瑚海的音乐风格是什么")
    # print(res)
    # res = graph.query("青花瓷是在哪一年发行的")
    # print(res)
    res = graph.query("周杰伦的粉丝名是什么")
    print(res)
    res = graph.query("周杰伦都使用那些社交平台")
    print(res)


