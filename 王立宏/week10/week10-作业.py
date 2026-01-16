# coding:utf8

'''
使用Bert完成自回归语言模型训练
'''

import torch
import torch.nn as nn
import numpy as np
import math, random, os, re
from transformers import BertTokenizer, BertModel


class LanguageModel(nn.Module):
    """
    基于BERT的自回归语言模型
    通过添加因果注意力掩码，将BERT的双向注意力改为单向注意力
    """
    def __init__(self, hidden_size, vocab_size, pretrain_model_path):
        """
        初始化语言模型
        :param hidden_size: BERT隐藏层大小（768）
        :param vocab_size: 词表大小
        :param pretrain_model_path: 预训练模型路径
        """
        super(LanguageModel, self).__init__()
        # 加载预训练BERT模型
        # return_dict=False: 返回元组格式而非字典
        # attn_implementation='eager': 使用标准注意力实现（非优化版本）
        self.bert = BertModel.from_pretrained(pretrain_model_path,
                                              return_dict=False,
                                              attn_implementation='eager')
        # 分类层：将BERT的768维隐藏状态映射到词表大小的logits
        self.classify = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, y=None, attention_mask=None):
        """
        前向传播
        :param x: 输入token ids [batch_size, seq_len]
        :param y: 目标token ids（训练时使用）[batch_size, seq_len]
        :param attention_mask: 注意力掩码（推理时使用）
        :return: 训练时返回loss，推理时返回预测概率
        """
        if y is not None:
            # 训练模式：需要计算loss
            # 创建下三角因果掩码（causal mask）
            # 位置(i,j)只能看到j<=i的信息，确保自回归特性
            mask = torch.tril(torch.ones(x.size(1), x.size(1))).unsqueeze(0).expand(x.size(0), -1, -1)
            if x.is_cuda: mask = mask.cuda()
            # 将掩码传入BERT，确保每个位置只能关注之前的位置
            x, _ = self.bert(x, attention_mask=mask)
            # 通过分类层得到logits [batch_size, seq_len, vocab_size]
            logits = self.classify(x)
            # 计算交叉熵损失
            # view(-1, ...) 将batch和seq维度展平，计算每个位置的预测loss
            return nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        else:
            # 推理模式：只需要预测概率
            x, _ = self.bert(x, attention_mask=attention_mask)
            logits = self.classify(x)
            # 对vocab维度做softmax，得到每个token的概率分布
            return torch.softmax(logits, dim=-1)


def load_corpus(path):
    """
    加载语料文件
    :param path: 语料文件路径
    :return: 连接后的完整文本字符串
    """
    with open(path, encoding='gbk') as f:
        # 读取所有行并去除首尾空白字符，然后拼接成一个完整字符串
        return ''.join([line.strip() for line in f])


def build_sample(tokenizer, window_size, corpus):
    """
    构建单个训练样本
    从语料中随机采样一段文本，构建输入和目标
    :param tokenizer: 分词器
    :param window_size: 窗口大小（序列长度）
    :param corpus: 完整语料文本
    :return: 输入token ids和目标token ids
    """
    # 随机选择起始位置（留出window_size+1的空间，因为需要预测下一个token）
    start = random.randint(0, len(corpus) - window_size - 1)
    # 输入窗口：从start开始的window_size个字符
    window = corpus[start: start + window_size]
    # 目标窗口：从start+1开始的window_size个字符（错位1位，用于预测下一个token）
    target = corpus[start + 1: start + window_size + 1]
    # 将输入文本编码为token ids
    x = tokenizer.encode(window, add_special_tokens=False,
                         padding='max_length', truncation=True, max_length=window_size)
    # 将目标文本编码为token ids
    y = tokenizer.encode(target, add_special_tokens=False,
                         padding='max_length', truncation=True, max_length=window_size)
    return x, y


def build_dataset(sample_num, tokenizer, window_size, corpus):
    """
    构建训练数据集
    :param sample_num: 样本数量
    :param tokenizer: 分词器
    :param window_size: 窗口大小
    :param corpus: 完整语料文本
    :return: 输入张量和目标张量
    """
    xs, ys = [], []
    # 循环构建样本
    for _ in range(sample_num):
        x, y = build_sample(tokenizer, window_size, corpus)
        xs.append(x);
        ys.append(y)
    # 转换为PyTorch张量
    return torch.LongTensor(xs), torch.LongTensor(ys)


def top_k_sampling(prob, k=30, temp=1.0):
    """
    Top-K采样策略
    从概率最高的k个token中按概率采样，避免只选择最高概率的token，增加多样性
    :param prob: 概率分布 [vocab_size]
    :param k: 保留前k个最高概率的token
    :param temp: 温度参数，越大概率分布越平滑，越小越集中
    :return: 采样得到的token id
    """
    # 获取概率最高的k个token及其概率
    prob, idx = torch.topk(prob, k)
    # 应用温度参数并重新归一化（温度越高，分布越均匀）
    prob = torch.softmax(prob / temp, dim=-1)
    # 从k个候选中按概率采样1个
    return idx[torch.multinomial(prob, 1)].item()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_len=50):
    """
    文本生成函数
    使用自回归方式逐步生成文本
    :param model: 语言模型
    :param tokenizer: 分词器
    :param prompt: 提示文本（起始文本）
    :param max_len: 最大生成长度
    :return: 生成的完整文本
    """
    model.eval()  # 设置为评估模式
    device = next(model.parameters()).device  # 获取模型所在设备
    # 将提示文本编码为token ids
    ids = tokenizer.encode(prompt, add_special_tokens=False)
    # 自回归生成循环
    for _ in range(max_len):
        # 将当前序列转换为张量 [1, seq_len]
        x = torch.LongTensor([ids]).to(device)
        seq_len = x.size(1)
        # 创建因果掩码（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).to(device)
        # 前向传播，获取最后一个位置的预测概率 [vocab_size]
        y = model(x, attention_mask=mask)[0, -1]
        # 使用top-k采样选择下一个token
        nid = top_k_sampling(y, k=30, temp=0.9)
        ids.append(nid)  # 将新token添加到序列中
        # 如果生成了结束符，则停止生成
        if nid == tokenizer.sep_token_id:
            break

    # 将token ids解码为文本，移除特殊token并去除空格
    text = tokenizer.decode(ids, skip_special_tokens=True).replace(' ', '')
    return text


def train(corpus_path, save_weight=True):
    """
    训练函数
    :param corpus_path: 语料文件路径
    :param save_weight: 是否保存模型权重
    """
    # ========== 超参数设置 ==========
    epoch_num = 20              # 训练轮数
    batch_size = 128            # 批次大小
    train_sample = 10000        # 每轮训练样本总数
    window_size = 32            # 序列窗口大小
    learning_rate = 1e-4        # 学习率
    pretrain_path = r'./bert-base-chinese'  # 预训练模型路径

    # ========== 初始化 ==========
    # 加载分词器
    tokenizer = BertTokenizer.from_pretrained(pretrain_path)
    # 加载语料
    corpus = load_corpus(corpus_path)
    # 创建模型（hidden_size=768是bert-base的隐藏层大小）
    model = LanguageModel(768, tokenizer.vocab_size, pretrain_path)
    # 将模型移动到GPU（如果可用）
    if torch.cuda.is_available():
        model = model.cuda()
    # 创建优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # ========== 开始训练 ==========
    print('开始训练……')
    for epoch in range(epoch_num):
        model.train()  # 设置为训练模式
        watch_loss = []  # 记录每个batch的loss
        # 每轮训练 train_sample // batch_size 个batch
        for _ in range(train_sample // batch_size):
            # 构建一个批次的数据
            x, y = build_dataset(batch_size, tokenizer, window_size, corpus)
            # 数据移到GPU
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            # 前向传播
            optim.zero_grad()  # 梯度清零
            loss = model(x, y)  # 计算loss
            # 反向传播
            loss.backward()  # 计算梯度
            optim.step()  # 更新参数
            # 记录loss
            watch_loss.append(loss.item())
        # 计算平均loss和困惑度（perplexity）
        avg_loss = np.mean(watch_loss)
        ppl = math.exp(avg_loss)  # 困惑度 = exp(cross_entropy_loss)
        print(f'第{epoch + 1}轮  loss={avg_loss:.4f}  ppl={ppl:.2f}')
        # 每轮结束后生成示例文本，观察训练效果
        print(generate(model, tokenizer, '让他在半年之前，就不能做出'))
        print(generate(model, tokenizer, '李慕站在山路上，深深的呼吸'))
    # ========== 保存模型 ==========
    if save_weight:
        os.makedirs('model', exist_ok=True)  # 创建保存目录
        # 根据语料文件名生成保存路径
        save_path = os.path.join('model', os.path.basename(corpus_path).replace('.txt', '.pth'))
        torch.save(model.state_dict(), save_path)  # 保存模型权重
        print('权重已保存至', save_path)


if __name__ == '__main__':
    # 程序入口：启动训练
    # corpus.txt: 训练语料文件
    # save_weight=True: 训练完成后保存模型权重
    train(r'corpus.txt', save_weight=True)
