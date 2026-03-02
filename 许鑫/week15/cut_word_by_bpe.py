import json


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids, pair, idx):
    newids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            newids.append(idx)
            i += 2
        else:
            newids.append(ids[i])
            i += 1
    return newids


def bpe_data(path):
    vocab_size = 350
    num_merges = vocab_size - 256

    with open(path, encoding='utf-8') as f:
        data = f.read()
        f.close()
    tokens = data.encode('utf-8')
    tokens = list(map(int, tokens))
    ids = list(tokens)

    merges = {}  # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key=stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge(ids, pair, idx)
        merges[pair] = idx

    print("tokens length:", len(tokens))
    print("ids length:", len(ids))
    print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
        vocab[idx] = vocab[p0] + vocab[p1]
        print(idx, vocab[idx].decode("utf-8", errors="replace"))
    return merges, vocab


def decode(vocab, ids):
    # given ids (list of integers), return Python string
    tokens = b"".join(vocab[idx] for idx in ids)
    text = tokens.decode("utf-8", errors="replace")
    return text


def encode(merges, text):
    # given a string, return list of integers (the tokens)
    tokens = list(text.encode("utf-8"))
    while len(tokens) >= 2:
        stats = get_stats(tokens)
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break  # nothing else can be merged
        idx = merges[pair]
        tokens = merge(tokens, pair, idx)
    return tokens


if __name__ == '__main__':
    merges, vocab = bpe_data('./data/tag_news.json')
    # demo_text = '北京东富西贵'
    demo_text = '为逃避法律追究，他竟造假遗书说是自杀殉情'
    demo_text_tokens = encode(merges, demo_text)
    print('-' * 80)
    print(demo_text_tokens)
    print('-' * 80)
    demo_text_decode = decode(vocab, demo_text_tokens)
    print(demo_text_decode)
    print('-' * 80)
