# -*- coding: utf-8 -*-
import sys
import torch
import os
import random
import os
import numpy as np
import time
import logging
import json
from config import Config
# from evaluate import Evaluator
from loader import load_data
from loader import load_vocab
from model import TorchModel

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


"""
模型训练主程序
"""

# seed = Config["seed"]
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=learning_rate)


def main(config):
    window_size = 10
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载模型
    logger.info(json.dumps(config, ensure_ascii=False, indent=2))
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    #加载效果测试类
    vocab = load_vocab(config["bert_path"])

    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for index, batch_data in enumerate(train_data):
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, predict_seq,mask = batch_data

            loss = model(input_seq, predict_seq,mask)

            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info("epoch average loss: %f" % np.mean(train_loss))
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(train_loss)))
    print("美国的花岗岩和牡蛎的奇妙缘分--",generate_sentence("美国的花岗岩和牡蛎的奇妙缘分", model, vocab, window_size,config["output_max_length"]),config["output_max_length"])
    print("美国大选：川普大获全胜--",generate_sentence("美国大选：川普大获全胜", model, vocab, window_size,config["output_max_length"]),config["output_max_length"])
    print("艺术品金融如何持续发展--",generate_sentence("艺术品金融如何持续发展？", model, vocab, window_size,config["output_max_length"]),config["output_max_length"])
    print("各路足坛名人陆续抵达--",generate_sentence("各路足坛名人陆续抵达", model, vocab, window_size,config["output_max_length"]),config["output_max_length"])
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return

def generate_sentence1(openings, model, vocab, window_size,max_len):
    ask_size = len(openings)
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
        x=padding(vocab,x, max_len)
        x = torch.LongTensor([x])
        if torch.cuda.is_available():
            x = x.cuda()
        y = model(x)[0]
        # print(y.shape)
        pred_char = ""
        for ls in y:
            index=sampling_strategy(ls)
            pred_char += reverse_vocab[index]
    return pred_char[ask_size:]

def generate_sentence(openings, model, vocab, window_size,max_len):
    ask_size = len(openings)
    reverse_vocab = dict((y, x) for x, y in vocab.items())
    model.eval()
    with torch.no_grad():
        pred_char = ""
        #生成了换行符，或生成文本超过max_len字则终止迭代
        while pred_char != "\n" and len(openings) <= max_len:
            openings += pred_char
            x = [vocab.get(char, vocab["[UNK]"]) for char in openings[-window_size:]]
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            pred_char = reverse_vocab[index]

    return openings[ask_size:]
def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)
def padding(vocab, input_id, length):
    input_id = input_id[:length]
    input_id += [vocab["[PAD]"]] * (length - len(input_id))
    return input_id
if __name__ == "__main__":

    main(Config)


