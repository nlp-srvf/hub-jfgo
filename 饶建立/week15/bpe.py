import os
from typing import List, Tuple, Dict

"""
Byte Pair Encoding (BPE) 算法实现
用于构建子词词表（Subword Vocabulary），适用于大语言模型分词
"""


def get_stats(token_ids: List[int]) -> Dict[Tuple[int, int], int]:
    """
    统计相邻token对的出现频率

    Args:
        token_ids: token ID列表

    Returns:
        字典，key为相邻token对(tuple)，value为出现次数
    """
    counts = {}
    # 使用zip创建滑动窗口，统计相邻字符对
    for pair in zip(token_ids, token_ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(token_ids: List[int], pair: Tuple[int, int], new_idx: int) -> List[int]:
    """
    将所有指定的相邻token对合并为新的token

    Args:
        token_ids: 原始token序列
        pair: 需要合并的相邻token对
        new_idx: 合并后的新token ID

    Returns:
        合并后的新token序列
    """
    new_ids = []
    i = 0
    while i < len(token_ids):
        # 检查当前位置是否匹配要合并的pair
        if i < len(token_ids) - 1 and token_ids[i] == pair[0] and token_ids[i+1] == pair[1]:
            new_ids.append(new_idx)
            i += 2  # 跳过已合并的两个token
        else:
            new_ids.append(token_ids[i])
            i += 1
    return new_ids


def build_vocab(corpus: str, vocab_size: int = 500) -> Tuple[Dict, Dict]:
    """
    基于语料库训练BPE词表

    Args:
        corpus: 训练语料文本
        vocab_size: 目标词表大小（默认500）

    Returns:
        merges: 合并规则字典 {(int, int): int}
        vocab: 词表字典 {int: bytes}
    """
    print(f"开始构建词表，目标大小: {vocab_size}")

    num_merges = vocab_size - 256  # 基础字节0-255，需要新增的token数

    # Step 1: 文本编码为UTF-8字节序列
    text_bytes = corpus.encode("utf-8")
    token_ids = list(map(int, text_bytes))

    print(f"初始token数量: {len(token_ids)}")

    # Step 2: 迭代合并最频繁的token对
    merges = {}  # 存储合并规则

    for i in range(num_merges):
        stats = get_stats(token_ids)
        if not stats:
            break

        # 选择频率最高的pair进行合并
        best_pair = max(stats, key=stats.get)
        new_idx = 256 + i

        if i < 5 or i % 50 == 0:  # 只打印前5次和每50次，避免输出过多
            print(f"  第{i+1}次合并: {best_pair} → {new_idx} (频率: {stats[best_pair]})")

        token_ids = merge(token_ids, best_pair, new_idx)
        merges[best_pair] = new_idx

    print(f"合并后token数量: {len(token_ids)}")

    # Step 3: 构建可解码的词表
    vocab = {idx: bytes([idx]) for idx in range(256)}  # 基础字节表

    print("\n构建词表（部分示例）:")
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]  # 新token的字节表示

        # 尝试解码为可读字符（仅显示前20个作为示例）
        if idx < 276:
            try:
                char = vocab[idx].decode("utf-8")
                print(f"  Token {idx}: {repr(char)}")
            except UnicodeDecodeError:
                pass  # 部分token可能是无效UTF-8序列，这是正常的

    print(f"... 共 {len(merges)} 个合并规则")
    return merges, vocab


def encode(text: str, merges: Dict[Tuple[int, int], int]) -> List[int]:
    """
    使用BPE词表将文本编码为token序列

    Args:
        text: 待编码文本
        merges: 合并规则字典

    Returns:
        token ID列表
    """
    # 初始化为UTF-8字节
    token_ids = list(text.encode("utf-8"))

    # 迭代应用合并规则
    while len(token_ids) >= 2:
        stats = get_stats(token_ids)

        # 选择优先级最高（在merges中索引最小）的可合并pair
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))

        if pair not in merges:
            break  # 没有可合并的pair了

        new_idx = merges[pair]
        token_ids = merge(token_ids, pair, new_idx)

    return token_ids


def decode(token_ids: List[int], vocab: Dict[int, bytes]) -> str:
    """
    将token序列解码为文本

    Args:
        token_ids: token ID列表
        vocab: 词表字典

    Returns:
        解码后的字符串
    """
    # 拼接所有token对应的字节
    byte_seq = b"".join(vocab[idx] for idx in token_ids)
    # UTF-8解码，遇到错误用替换符
    text = byte_seq.decode("utf-8", errors="replace")
    return text


def load_corpus(dir_path: str) -> str:
    """
    从目录加载所有文本文件

    Args:
        dir_path: 文本文件所在目录

    Returns:
        合并后的语料文本
    """
    corpus = ""
    file_count = 0

    try:
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)

            # 只处理文件（跳过子目录）
            if os.path.isfile(file_path):
                try:
                    with open(file_path, encoding="utf-8") as f:
                        content = f.read()
                        corpus += content + "\n"
                        file_count += 1
                except Exception as e:
                    print(f"警告: 无法读取文件 {filename}: {e}")
    except FileNotFoundError:
        print(f"错误: 目录不存在: {dir_path}")
        return ""

    print(f"成功加载 {file_count} 个文件，语料总长度: {len(corpus)} 字符")
    return corpus


def main():
    """主函数：演示BPE训练、编码和解码流程"""

    # ========== 配置区域 ==========
    # 请修改为您的实际数据路径
    DATA_DIR = r"D:\Study_Mterials\BD_LLM\week14 大语言模型应用相关\week14 大语言模型应用相关\RAG"

    # 超参数设置
    VOCAB_SIZE = 500  # 词表大小，可根据语料规模调整

    # 测试文本
    TEST_STRINGS = [
        "蝙蝠骑士",
        "石鳞剑士",
        "水晶室女",
        "这是一个测试句子"
    ]
    # ==============================

    print("=" * 50)
    print("BPE (Byte Pair Encoding) 算法演示")
    print("=" * 50)

    # 1. 加载语料
    print("\n[1/4] 加载训练语料...")
    corpus = load_corpus(DATA_DIR)

    if not corpus:
        print("语料加载失败，使用示例文本进行演示")
        corpus = "石鳞剑士擅长近战。水晶室女拥有强大的魔法能力。"

    # 2. 训练BPE词表
    print("\n[2/4] 训练BPE词表...")
    merges, vocab = build_vocab(corpus, vocab_size=VOCAB_SIZE)
    print(f"词表构建完成！基础token: 256个，合并token: {len(merges)}个")

    # 3. 测试编码和解码
    print("\n[3/4] 测试编码和解码...")
    print("-" * 50)

    for text in TEST_STRINGS:
        print(f"\n原始文本: {text}")

        # 编码
        encoded = encode(text, merges)
        print(f"编码结果: {encoded}")
        print(f"Token数量: {len(encoded)} (原文字节数: {len(text.encode('utf-8'))})")

        # 解码
        decoded = decode(encoded, vocab)
        print(f"解码结果: {decoded}")

        # 验证
        if decoded == text:
            print("✓ 编解码验证通过")
        else:
            print("✗ 编解码不一致（可能有特殊字符）")

    # 4. 保存词表（可选）
    print("\n[4/4] 保存词表...")
    try:
        import json
        # 转换tuple key为string以便JSON序列化
        merges_serializable = {f"{k[0]},{k[1]}": v for k, v in merges.items()}
        vocab_serializable = {k: v.decode("utf-8", errors="replace") for k, v in vocab.items()}

        with open("bpe_merges.json", "w", encoding="utf-8") as f:
            json.dump(merges_serializable, f, ensure_ascii=False, indent=2)
        print("合并规则已保存到: bpe_merges.json")

        with open("bpe_vocab.json", "w", encoding="utf-8") as f:
            json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
        print("词表已保存到: bpe_vocab.json")
    except Exception as e:
        print(f"保存失败: {e}")

    print("\n" + "=" * 50)
    print("演示完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()