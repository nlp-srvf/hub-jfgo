

class MyTokenizer():
    def __init__(self,vocab_size,corpus_path):
        self.vocab = {}
        self.merges={}
        self.vocab_size = vocab_size  # the desired final vocabulary size  超参数：预期的最终词表大小，根据实际情况自己设置，大的词表会需要大的embedding层
        self.num_merges = self.vocab_size - 256
        self.corpus_path = corpus_path
        self.tokens=self.load_corpus()
        self.tokens2=[]
        self.get_vocab()

    def get_stats(self,ids):
        counts = {}
        for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def merge(self,ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def load_corpus(self,):
        tokens = []
        with open(self.corpus_path, encoding="utf8") as f:
            for line in f:
                line_tokens = line.encode("utf-8")
                tokens.extend(list(map(int, line_tokens)))
        return tokens


    def get_vocab(self):
        tokens = self.tokens
        ids = list(tokens)  # copy so we don't destroy the original list
        merges = {} # (int, int) -> int
        for i in range(self.num_merges-1):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            #print(f"merging {pair} into a new token {idx}")
            ids = self.merge(ids, pair, idx)
            merges[pair] = idx

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        self.merges=merges


    def decode(self,ids):
        # given ids (list of integers), return Python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def encode(self,text):
        # given a string, return list of integers (the tokens)
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stats(tokens)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break  # nothing else can be merged
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens


if __name__ == '__main__':
    mtk = MyTokenizer(512,"财经.txt")
    tokens = mtk.encode("年原油上涨到每桶147美元，是因为美国石油储备丰富，包括没有开采的以及大量的战略储备等，石油价格上涨符合美国的战略意图。而目前，宏源期货：粮油市场在政策的调控或预期下震荡走弱的可能性较大。美豆丰产压力、两节备货需求结束、大豆轮储等因素具有利空影响，充裕的流动性虽是利多因素但已受到了国家的关注。从盘面上看，豆类昨日放量增仓下行，下跌可能性较豆粕、豆油大。总体而言，我们认为大豆近日震荡走弱的可能较大，建议投资者观望或短空，豆粕表现较强，不宜轻易放空。白糖迈科期货：目前1105已于前高压力区整理多日，但在美糖破前高背景下期价仍没有明显上冲举动，昨日反出现较大幅度回落，加之华商储备竞拍白砂糖会多少扰乱市场参与者做多心神，故需要警惕各因素叠加导致的短期回调的出现。操作建议：多单谨慎持有，盘中冲高减持PVC象屿期货：周二国内商品走势疲弱，虽然隔夜外")
    print(tokens)
    print("========================len:",len(tokens))
    print(mtk.decode(tokens))




