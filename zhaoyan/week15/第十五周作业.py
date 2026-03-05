----------------------------------------------1BPE算法了解----------------------------------------------------------------------
现代主流大模型的词表，几乎都是基于“先将文本转换为UTF-8字节流，再运用BPE算法进行合并”这一范式构建的。BPE构建词表的过程，本质上是一个只依赖于文本（编码过程）的统计分析过程，完全不涉及解码过程。

总结一下BPE的逻辑，分为训练词表和使用词表编解码
1.训练词表
①将所有文本（字符）转化为多个（不超过4个）的16进制字节
② 迭代合并（你困惑的核心）
a统计所有相邻符号对的频率。
b合并频率最高的一对。
c记录此次合并的“符号对”到 merges.txt。
d. 在语料中，将所有出现该符号对的地方替换为新符号。
重复步骤2，直到达到预设的合并次数（词表大小 = 256 + 合并次数）。
③训练最终产出两个核心文件：
合并规则表 (merges.txt)：按顺序记录所有合并的符号对。
最终词表 (vocab.json)：一个包含所有符号（初始字节+合并出的新符号）及其唯一ID的双向映射表。每个符号对应一个唯一的字节序列。例如："256": "b'\x6c\x6f\x77'",但为了人们理解，通常在词表是"256": "我'"，使用了再转化

2.使用词表进行编解码
编码（字节化→贪婪最大匹配→ID 映射）：
①字节化：将输入文本转换为UTF-8字节序列。
②分词与映射ID：使用训练好的最终词表，即token_to_id 部分，在字节序列上执行贪婪的最大匹配（找到词表中存在的、最长的字节片段），将其切分成一系列“符号”（Tokens），将每个符号映射为对应的Token ID
解码（ID→字节串→拼接→UTF-8 解码）：
①查表映射：直接词表的 id_to_token 部分，将ID映射回其存储的字节串：256 -> b'\x6c\x6f\x77
②拼接字节：得到完整字节串
③UTF-8解码：将完整字节串交给UTF-8解码器，得到最终字符串 “lower”。

3.词表分词器结构及内容
# 这是分词器内部的核心数据结构概念
①token_to_id = {
b'\x00': 0,
b'\x0a': 10,
b'Hello': 1000,
b' world': 1001,
b'\xe4\xbd\xa0\xe5\xa5\xbd': 2000, # “你好”的字节
# ... 成千上万个条目
}
②id_to_token = {
0: b'\x00',
10: b'\x0a',
1000: b'Hello',
1001: b' world',
2000: b'\xe4\xbd\xa0\xe5\xa5\xbd',
# ... 反向映射
}
③词表：它用字符串形式存储Token，目的是让人和文本编辑器能够阅读、审查。
分词器在初始化时，会立刻将下面文件加载，并把每个Token的字符串转换为对应的UTF-8字节序列，形成内存中真正用于编解码的映射表即token_to_id。
{
"1": "<s>", // 句子开始符
"2": "</s>", // 句子结束符
"3": "<0x00>", // 基础字节 0x00
"4": "<0x01>",
"...": "...",
"258": "<0x0A>", // 换行符字节
"...": "...",
"1000": "Hello",
"1001": " world",
"1500": "def",
"1501": " main",
"1502": "():",
"2000": "你好",
"2001": "中国",
"...": "...",
"32000": "The",
"32001": " the",
"32002": " in",
"32003": " and",
"...": "..."
}
4.合并规则表 merges.txt 文件说明
self.merges = {
    (230, 136): 256,      # 第一次合并
    (256, 145): 257,      # 第二次合并
    (231, 136): 258,      # 第三次合并
    # ... 继续记录所有合并
}
注：在大多数现代的、用于生产环境的“编码/解码”使用过程中，merges.txt 文件本身确实不再被直接使用，但它仍然重要。
①对于已加载的、运行中的分词器：没用
当分词器（如 GPT2Tokenizer 或 LlamaTokenizer）被成功加载到内存中后，其核心的 token_to_id 和 id_to_token 字典已经构建完毕。此时进行编码或解码：
编码：使用 token_to_id 字典进行贪婪匹配，不读取 merges.txt。
解码：使用 id_to_token 字典进行查找拼接，也不读取 merges.txt。
在这个运行阶段，merges.txt 是不被访问的，可以被视为“离线”。
②对于分词器的初始化和兼容性：有用且必要
虽然运行时不用，但在加载分词器的那一刻，merges.txt 可能至关重要：
场景A：经典分体式文件（如原始GPT-2）
如果你只有 vocab.json （ID到字符串的映射）而没有 merges.txt，很多库无法正确构建出用于高效编码的 token_to_id 字典。因为 vocab.json 只记录了最终结果，而 merges.txt 记录了得到这些结果的“合并规则”，某些初始化逻辑需要利用这些规则来确保分词器行为与训练时完全一致（尤其是在处理那些需要按顺序合并的边缘情况时）。


----------------------------------------------2代码----------------------------------------------------------------------
注：我的教学代码例子中，词表训练好后，编码没有用训练好的词表，用了self.merges，而实际编码大模型是不用self.merges。
我的代码例子（教学实现）和实际大模型（生产实现）有两个根本区别：
特性	      我的代码例子（教学实现）	                         实际大模型（生产实现）
1. 编码机制	使用self.merges 动态模拟合并过程	                 使用固化词表直接进行最大匹配
2. 词表使用	只有基础的self.vocab映射，没有预编译的最终匹配词表	 有专门的token_to_id映射用于高效查找
3. 数据结构	编码时需动态统计和查找merges	使用Trie树（前缀树）   实现O(n)匹配
4. 效率	    极低，每次编码都重走训练过程	                        极高，一次扫描完成分词

import re
from collections import defaultdict
import jieba
import matplotlib.pyplot as plt

# ==================== 1. 准备语料 ====================
corpus = """
却说那女娲氏炼石补天之时，于大荒山无稽崖炼成高十二丈、见方二十四丈大的顽石三万六千五百零一块。
那娲皇只用了三万六千五百块，单单剩下一块未用，弃在青埂峰下。
谁知此石自经锻炼之后，灵性已通，自去自来，可大可小。
因见众石俱得补天，独自己无才不得入选，遂自怨自愧，日夜悲哀。
一日，正当嗟悼之际，俄见一僧一道远远而来，生得骨格不凡，丰神迥异，
说说笑笑来至峰下，坐于石边高谈快论。
先是说些云山雾海神仙玄幻之事，后便说到红尘中荣华富贵。
此石听了，不觉打动凡心，也想要到人间去享一享这荣华富贵，
但自恨粗蠢，不得已，便口吐人言，向那僧道说道：
"大师，弟子蠢物，不能见礼了。适闻二位谈那人世间荣耀繁华，
心切慕之。弟子质虽粗蠢，性却稍通，况见二师仙形道体，定非凡品，
必有补天济世之材，利物济人之德。如蒙发一点慈心，携带弟子得入红尘，
在那富贵场中、温柔乡里受享几年，自当永佩洪恩，万劫不忘也。"
二仙师听毕，齐憨笑道："善哉，善哉！那红尘中有却有些乐事，
但不能永远依恃，况又有'美中不足，好事多魔'八个字紧相连属，
瞬息间则又乐极悲生，人非物换，究竟是到头一梦，万境归空，
倒不如不去的好。"
这石凡心已炽，那里听得进这话去，乃复苦求再四。
二仙知不可强制，乃叹道："此亦静极思动，无中生有之数也。
既如此，我们便携你去受享受享，只是到不得意时，切莫后悔。"
石道："自然，自然。"那僧又道："若说你性灵，却又如此质蠢，
并更无奇贵之处，如此也只好踮脚而已。也罢，我如今大施佛法助你助。
待劫终之日，复还本质，以了此案。你道好否？"石头听了，感谢不尽。
"""

print(f"语料长度：{len(corpus)} 字符")
print("语料前200字：")
print(corpus[:200] + "...")


# ==================== 2. 预处理语料 ====================
def preprocess_text(text):
"""预处理文本：分词并添加分隔符"""
# 使用jieba进行中文分词
words = jieba.lcut(text)
# 在词之间添加空格作为分隔符
processed = " ".join(words)
return processed


processed_corpus = preprocess_text(corpus)
print("\n处理后的语料（已分词）：")
print(processed_corpus[:200] + "...")


# ==================== 3. BPE算法核心实现 ====================
class BPETokenizer:
def __init__(self, vocab_size=500):
self.vocab_size = vocab_size
self.vocab = {}
self.merges = {} # 存储合并规则
self.inverse_vocab = {} # 反向查找表

def get_stats(self, tokens):
"""统计相邻token对的出现频率"""
stats = defaultdict(int)
for pair in zip(tokens, tokens[1:]):
stats[pair] = stats.get(pair, 0) + 1
return stats

def merge(self, tokens, pair, idx):
"""合并指定的token对"""
new_tokens = []
i = 0
while i < len(tokens):
if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
new_tokens.append(idx)
i += 2
else:
new_tokens.append(tokens[i])
i += 1
return new_tokens

def train(self, text):
"""训练BPE分词器"""
# 初始token：所有字符
chars = list(text.encode('utf-8'))
tokens = chars.copy()

# 初始词汇表：0-255的字节
self.vocab = {i: bytes([i]) for i in range(256)}

# BPE训练
merges = {}
for i in range(self.vocab_size - 256):
stats = self.get_stats(tokens)
if not stats:
break

# 找到最频繁的pair
top_pair = max(stats, key=stats.get)

# 分配新token id
idx = 256 + i

# 执行合并
tokens = self.merge(tokens, top_pair, idx)

# 记录合并规则
merges[top_pair] = idx
self.merges[top_pair] = idx

# 更新词汇表
self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

if (i + 1) % 50 == 0:
print(f"第{i + 1}次合并：{top_pair} -> {idx}")
print(f"当前token序列长度：{len(tokens)}")

# 构建反向查找表
self.inverse_vocab = {v: k for k, v in enumerate(self.vocab)}

print(f"\n训练完成！词汇表大小：{len(self.vocab)}")
print(f"最终token序列长度：{len(tokens)}")
print(f"压缩率：{len(chars) / len(tokens):.2f}x")

return tokens

def encode(self, text):
"""编码文本为token序列"""
# 转换为字节序列
tokens = list(text.encode('utf-8'))

# 应用合并规则
while len(tokens) >= 2:
stats = self.get_stats(tokens)

# 找到可以合并的pair
pair_to_merge = None
for pair in stats:
if pair in self.merges:
if pair_to_merge is None or self.merges[pair] < self.merges.get(pair_to_merge, float('inf')):
pair_to_merge = pair

if pair_to_merge is None:
break

idx = self.merges[pair_to_merge]
tokens = self.merge(tokens, pair_to_merge, idx)

return tokens

def decode(self, tokens):
"""解码token序列为文本"""
# 将token转换为字节
bytes_data = b""
for token in tokens:
if token in self.vocab:
bytes_data += self.vocab[token]
else:
# 如果token不在词汇表中，使用原始字节
bytes_data += bytes([token])

# 解码为文本
try:
return bytes_data.decode('utf-8', errors='replace')
except:
return bytes_data.decode('gbk', errors='replace')


# ==================== 4. 训练BPE分词器 ====================
print("\n" + "=" * 50)
print("开始训练BPE分词器...")
print("=" * 50)

tokenizer = BPETokenizer(vocab_size=400)
tokens = tokenizer.train(processed_corpus)

# ==================== 5. 编码和解码测试 ====================
print("\n" + "=" * 50)
print("编码/解码测试")
print("=" * 50)

test_sentences = [
"却说那女娲氏炼石补天之时",
"此石听了，不觉打动凡心",
"那红尘中有却有些乐事",
"究竟是到头一梦，万境归空"
]

for i, sentence in enumerate(test_sentences):
processed = preprocess_text(sentence)
encoded = tokenizer.encode(processed)
decoded = tokenizer.decode(encoded)

print(f"\n测试句子 {i + 1}:")
print(f"原始: {sentence}")
print(f"分词后: {processed}")
print(f"编码后token数量: {len(encoded)}")
print(f"解码后: {decoded}")
print(f"是否一致: {processed == decoded}")
