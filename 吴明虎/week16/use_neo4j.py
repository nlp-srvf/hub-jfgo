import re
import json
from py2neo import Graph
from collections import defaultdict

'''
读取三元组，并将数据写入neo4j
'''


#连接图数据库
graph = Graph("http://localhost:7474",auth=("neo4j","huge"))

cypher = "MATCH (n) DETACH DELETE n"
graph.run(cypher)
# # 關閉連接
# driver.close()