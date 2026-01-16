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
from model import LanguageModel
from transformers import BertTokenizer

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

    model = LanguageModel(768, 21128, config["bert_path"])
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    # 加载训练数据
    train_data = load_data(config["train_data_path"], config, logger)
    # #加载效果测试类
    # vocab = load_vocab(config["bert_path"])

    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        if cuda_flag:
            model.cuda()
        logger.info("epoch %d begin" % epoch)
        train_loss = []
        for batch_data in train_data:
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]
            input_seq, mask ,predict_seq= batch_data

            loss = model(input_seq,mask,predict_seq)

            train_loss.append(float(loss))
            loss.backward()
            optimizer.step()

        logger.info("epoch average loss: %f" % np.mean(train_loss))
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
    print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(train_loss)))
    print(generate_sentence("北京明年拟推工作日半价观看电影", model, tokenizer,config["output_max_length"]))
    print(generate_sentence("南京一合金厂锅炉发生爆炸", model, tokenizer,config["output_max_length"]))
    model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    torch.save(model.state_dict(), model_path)
    return


#文本生成测试代码
def generate_sentence(openings, model, tokenizer,length):
    model.eval()
    openings = tokenizer.encode(openings)
    with torch.no_grad():
        #生成文本超过30字则终止迭代
        while len(openings) <= length:
            x = torch.LongTensor([openings])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)[0][-1]
            index = sampling_strategy(y)
            openings.append(index)
    return tokenizer.decode(openings)

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

if __name__ == "__main__":

    main(Config)


