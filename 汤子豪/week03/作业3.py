import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中某个特定字符的索引

"""

class TorchRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRNN, self).__init__()
        self.layer = nn.RNN(input_size, hidden_size, bias=True, batch_first=True)

    def forward(self, x):
        return self.layer(x)

class TorchModel(nn.Module):
    def __init__(self, vector_dim, vocab, sentence_length):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # 文本embedding层
        self.target_embedding = nn.Embedding(len(vocab), vector_dim)  # 目标字符嵌入层
        self.rnn = TorchRNN(2 * vector_dim, vector_dim)  # 用rnn提取文本与特定字符的特征
        self.classify = nn.Linear(vector_dim, sentence_length)  # 线性层
        self.loss = nn.CrossEntropyLoss()  # loss函数采用交叉熵损失
        self.sentence_length = sentence_length

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target_char, y=None):
        # 字符序列嵌入
        seq_embed = self.embedding(x)  # (batch_size, sen_len, vector_dim)
        # 目标字符嵌入并扩展
        target_embed = self.target_embedding(target_char)  # (batch_size, vector_dim)
        target_embed = target_embed.expand(-1, self.sentence_length,
                                                        -1)  # (batch_size, sen_len, vector_dim)
        # 拼接字符嵌入和目标字符嵌入
        combined = torch.cat([seq_embed, target_embed], dim=2)  # (batch_size, sen_len, vector_dim*2)
        # 输入RNN模型
        _, rnn_out = self.rnn(combined) # (batch_size, sen_len + 1, vector_dim) -> (batch_size, vector_dim)
        rnn_out = rnn_out.squeeze()
        # 输入分类层
        y_pred = self.classify(rnn_out) # (batch_size, vector_dim) -> (batch_size, sen_len)
        if y is not None:
            return self.loss(y_pred, y.squeeze().long())   # 预测值和真实值计算损失
        else:
            return torch.softmax(y_pred, dim=-1) # 输出预测结果

# 字符集随便挑了一些字，实际上还可以扩充
# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他她们是一或者几"  # 字符集
    vocab = {'[pad]': 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1   # 每个字对应一个序号
    vocab['[unk]'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(chars, vocab, sentence_length, num_embeddings):
    # 随机从字表选取sentence_length个字，可能重复
    x_string = ''.join(random.choices(chars, k=sentence_length))
    # 随机选择一个目标字符位置
    char_index = random.randint(0, sentence_length - 1)
    target_char = x_string[char_index]
    # 转换为索引
    seq_x = str_to_sequence(x_string, vocab, num_embeddings)
    seq_target = [vocab.get(target_char, vocab["[unk]"])]
    return seq_x, seq_target, char_index


# 建立数据集
def build_dataset(all_chars, sample_length, vocab, sentence_length, num_embeddings):
    dataset_x = []
    dataset_target = []
    dataset_y = []
    for _ in range(sample_length):
        x, target_char_idx, char_index = build_sample(all_chars, vocab, sentence_length, num_embeddings)
        dataset_x.append(x)
        dataset_target.append(target_char_idx)
        dataset_y.append(char_index)
    # 转换为张量
    x_tensor = torch.LongTensor(dataset_x)
    target_tensor = torch.LongTensor(dataset_target)
    y_tensor = torch.LongTensor(dataset_y)
    return x_tensor, target_tensor, y_tensor

def str_to_sequence(string, vocab, num_embeddings):
    seq = [vocab.get(s, vocab["[unk]"]) for s in string][:num_embeddings]
    if len(seq) < num_embeddings:
        seq += [vocab["[pad]"]] * (num_embeddings - len(seq))
    return seq

# 建立模型
def build_model(char_dim, vocab, sentence_length):
    model = TorchModel(char_dim, vocab, sentence_length)
    return model

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, all_chars, sample_length, vocab, sentence_length, num_embeddings):
    model.eval()
    x, target, y = build_dataset(all_chars, sample_length, vocab, sentence_length, num_embeddings)
    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x, target)  # (batch_size, sen_len, sentence_length)
        # 获取预测位置
        pred_positions = torch.argmax(y_pred, dim=1)
        # 计算准确率
        for i in range(len(y)):
            # 对于每个样本，找到目标字符实际位置
            actual_pos = y[i].item()
            pred_pos = pred_positions[i]
            # 检查预测位置是否包含实际位置
            if actual_pos == pred_pos:
                correct += 1
            total += 1
    acc = correct / total if total > 0 else 0
    print(f"正确预测个数: {correct}/{total}, 正确率: {acc:.4f}")
    return acc

def main():
    # 配置参数
    epoch_num = 30  # 训练轮数
    batch_size = 200  # 批次大小
    train_sample = 3000  # 训练样本总数
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    learning_rate = 0.001  # 学习率

    my_chars = "你我他她们是一或者几鱼肉"
    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(char_dim, vocab, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        total_loss = 0
        # 创建训练数据集
        x, target, y = build_dataset(my_chars, train_sample, vocab, sentence_length, sentence_length)
        # 批处理训练
        for i in range(0, train_sample, batch_size):
            end_idx = min(i + batch_size, train_sample)
            batch_x = x[i:end_idx]
            batch_target = target[i:end_idx]
            batch_y = y[i:end_idx]
            optim.zero_grad()
            loss = model(batch_x, batch_target, batch_y)
            loss.backward()
            optim.step()
            total_loss += loss.item()
        avg_loss = total_loss / (train_sample / batch_size)
        print(f"=========\n第{epoch + 1}轮平均loss:{avg_loss:.4f}, 学习率:{optim.param_groups[0]['lr']:.6f}")
        # 测试本轮模型结果
        acc = evaluate(model, my_chars, train_sample, vocab, sentence_length, sentence_length)
        log.append([acc, avg_loss])
    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    with open("vocab.json", "w", encoding="utf8") as writer:
        json.dump(vocab, writer, ensure_ascii=False, indent=2)
    return


#使用训练好的模型做预测
# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings, target_chars):
    char_dim = 64  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    # 加载字符表
    with open(vocab_path, "r", encoding="utf8") as f:
        vocab = json.load(f)
    # 建立模型
    model = build_model(char_dim, vocab, sentence_length)
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))
    # 处理输入
    x_list = []
    target_list = []
    for input_string, target_char in zip(input_strings, target_chars):
        # 将输入序列化，不足长度补0
        seq = [vocab.get(char, vocab['[unk]']) for char in input_string]
        if len(seq) < sentence_length:
            seq += [vocab['[pad]']] * (sentence_length - len(seq))
        x_list.append(seq[:sentence_length])
        # 目标字符索引
        target_idx = vocab.get(target_char, vocab['[unk]'])
        target_list.append([target_idx])
    # 转换为张量
    x_tensor = torch.LongTensor(x_list)
    target_tensor = torch.LongTensor(target_list)
    # 预测
    model.eval()
    with torch.no_grad():
        result = model(x_tensor, target_tensor)  # (batch_size, sen_len, sentence_length)
    # 获取预测位置
    pred_positions = torch.argmax(result, dim=-1)
    # 打印结果
    for i, (input_string, target_char) in enumerate(zip(input_strings, target_chars)):
        print(f"输入文本：{input_string}, 目标字符：{target_char}")
        # 实际位置
        actual_positions = []
        for j, char in enumerate(input_string[:sentence_length]):
            if char == target_char:
                actual_positions.append(j)
        # 预测位置
        pred = pred_positions[i]
        print(f"  预测位置：{pred}")
        print(f"  实际位置：{actual_positions}")
        # 检查预测是否包含实际位置
        if actual_positions:
            correct = any(pos == pred for pos in actual_positions)
            print(f"  预测结果：{'正确' if correct else '错误'}")
        else:
            print(f"  目标字符不在文本中")


if __name__ == "__main__":
    main()
    # test_strings = ["你我他", "她们是一或者", "你们是一个"]
    # test_target = ['我', '一', '们']
    # predict("model.pth", "vocab.json", test_strings, test_target)
