class BPE:
    def __init__(self):
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.merges = {}

    @staticmethod
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    @staticmethod
    def merge(ids, pair, new_idx):
        new_ids = []
        i = 0
        n = len(ids)  # 提前取长度，减少循环内的属性查找
        while i < n:
            if i < n - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def _update_stats(self, counts, ids, pair, new_idx):
        """辅助方法：合并后动态更新词频，仅修改受影响的词对"""
        i = 0
        while i < len(ids):
            if ids[i] == new_idx:
                # 处理新token的左邻词对：左元素+新token
                if i > 0:
                    left = ids[i - 1]
                    old_p = (left, pair[0])  # 合并前的旧词对
                    if old_p in counts:
                        counts[old_p] -= 1
                        if counts[old_p] == 0:
                            del counts[old_p]
                    new_p = (left, new_idx)  # 合并后的新词对
                    counts[new_p] = counts.get(new_p, 0) + 1
                # 处理新token的右邻词对：新token+右元素
                if i < len(ids) - 1:
                    right = ids[i + 1]
                    old_p = (pair[1], right)  # 合并前的旧词对
                    if old_p in counts:
                        counts[old_p] -= 1
                        if counts[old_p] == 0:
                            del counts[old_p]
                    new_p = (new_idx, right)  # 合并后的新词对
                    counts[new_p] = counts.get(new_p, 0) + 1
                i += 1
            else:
                i += 1

    def train(self, text, vocab_size=500):
        if vocab_size < 256:
            raise ValueError("vocab_size必须≥256")
        tokens = list(text.encode("utf-8"))
        ids = tokens.copy()
        num_merges = vocab_size - 256
        print(f"开始训练：基础字节256个，需合并{num_merges}次，目标词表{vocab_size}")

        # 核心优化1：仅初始化1次词频，不再重复调用get_stats
        counts = self.get_stats(ids)

        for i in range(num_merges):
            if not counts:  # 无可用词对则提前终止
                break
            top_pair = max(counts, key=counts.get)
            new_idx = 256 + i
            # 合并ids（轻微优化后，循环内少一次len调用）
            ids = self.merge(ids, top_pair, new_idx)
            # 记录合并规则和词表
            self.merges[top_pair] = new_idx
            self.vocab[new_idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
            # 核心优化2：动态更新词频，仅修改受影响的词对
            self._update_stats(counts, ids, top_pair, new_idx)
            # 可选：减少打印次数，降低IO耗时（IO比内存计算慢得多）
            if (i + 1) % 200 == 0:
                print(f"已合并{i + 1}/{num_merges}次，高频对：{top_pair}→{new_idx}")
        print("训练完成！")
        return self.merges, self.vocab

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_idx = self.merges[pair]
            tokens = self.merge(tokens, pair, new_idx)
        return tokens

    def decode(self, ids):
        bytes_seq = b"".join([self.vocab[idx] for idx in ids if idx in self.vocab])
        return bytes_seq.decode("utf-8", errors="replace")


if __name__ == "__main__":
    try:
        # 保留你原有的读取逻辑，若仍报编码错，替换为之前的自动识别编码即可
        with open("corpus.txt", "r", encoding="utf-8") as f:
            train_text = f.read()
        print(f"成功读取文档，文本长度：{len(train_text)}字符")
    except FileNotFoundError:
        print("错误：未找到corpus.txt文件，请确保文件在代码同级目录")
        exit()
    except UnicodeDecodeError:
        print("错误：文件编码不是UTF-8，建议替换为自动识别编码的读取逻辑")
        exit()

    bpe = BPE()
    bpe.train(train_text, vocab_size=1000)  # 调用逻辑完全不变

    test_texts = [
        "主力合约突破21000元/吨重要关口",
        "王羲之《平安帖》成交价达3.08亿元",
        "北京世茂宫园地处东三环CBD核心位置",
        "红木家具价格同比上涨60%，游资炒作明显"
    ]

    for text in test_texts:
        ids = bpe.encode(text)
        decoded = bpe.decode(ids)
        print(f"\n原始：{text}")
        print(f"编码Token：{ids[:15]}...")
        print(f"解码：{decoded}")
        print(f"一致性：{text == decoded}")