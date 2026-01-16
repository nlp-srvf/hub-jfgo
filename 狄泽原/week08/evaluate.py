# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():
            question_matrixs = torch.stack(self.question_ids, dim=0)
            if torch.cuda.is_available():
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)
            #将所有向量都作归一化 v / |v|
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            anchor , positive, negative , _ = batch_data
            # input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况

            with torch.no_grad():
                self._eval_triplet_matching(anchor, positive, negative)
                anchor_labels = self._get_anchor_labels(anchor)
                self._eval_knwb_matching(anchor, anchor_labels)

            # 打印评估结果
            self.show_stats()
            # 返回核心评估指标（用于保存最优模型）
            return self.stats_dict["correct"] / (self.stats_dict["correct"] + self.stats_dict["wrong"])

    def _eval_knwb_matching(self, test_vectors, true_labels):
        """原有逻辑：验证测试样本与知识库的匹配准确率"""
        # 归一化测试向量
        test_vectors = torch.nn.functional.normalize(test_vectors, dim=-1)
        # 计算与知识库所有向量的相似度
        sim_matrix = torch.mm(test_vectors, self.knwb_vectors.T)  # [batch_size, knwb_size]
        # 找到相似度最高的索引
        hit_indices = torch.argmax(sim_matrix, dim=1)

        # 统计正确/错误数
        for hit_idx, true_label in zip(hit_indices, true_labels):
            hit_label = self.question_index_to_standard_question_index[int(hit_idx)]
            if int(hit_label) == int(true_label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1

    def _get_anchor_labels(self, anchor, batch_idx):
        """获取Anchor样本的真实标签（适配不同测试集格式）"""
        # 方式1：若测试集三元组第四列是标签索引张量（推荐）
        # 需修改load_data的测试集加载逻辑，将标签索引存入batch的第四个元素
        # 示例：anchor_labels = batch_data[3][:, 0]  # 假设标签张量是[batch_size, 2]

        # 方式2：从知识库反向匹配（兼容无标签的情况，精度略低）
        anchor_vec = torch.nn.functional.normalize(anchor, dim=-1)
        sim_matrix = torch.mm(anchor_vec, self.knwb_vectors.T)  # [batch_size, knwb_size]
        hit_indices = torch.argmax(sim_matrix, dim=1)  # [batch_size]
        anchor_labels = [self.question_index_to_standard_question_index[int(idx)] for idx in hit_indices]
        return torch.LongTensor(anchor_labels).to(anchor.device)
    def _eval_triplet_matching(self, anchor_vec, positive_vec, negative_vec):
        """评估三元组匹配：Anchor-Positive相似度 > Anchor-Negative相似度"""
        # 归一化向量
        anchor_vec = torch.nn.functional.normalize(anchor_vec, dim=-1)
        positive_vec = torch.nn.functional.normalize(positive_vec, dim=-1)
        negative_vec = torch.nn.functional.normalize(negative_vec, dim=-1)

        # 计算余弦相似度（越高越相似）
        sim_ap = torch.sum(anchor_vec * positive_vec, dim=-1)  # [batch_size]
        sim_an = torch.sum(anchor_vec * negative_vec, dim=-1)  # [batch_size]

        # 统计正确数：sim_ap > sim_an
        correct_mask = sim_ap > sim_an
        self.stats_dict["triplet_correct"] += correct_mask.sum().item()
        self.stats_dict["triplet_total"] += len(anchor_vec)
    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)
        for test_question_vector, label in zip(test_question_vectors, labels):
            #通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            #test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            hit_index = int(torch.argmax(res.squeeze())) #命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index] #转化成标准问编号
            if int(hit_index) == int(label):
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        return

    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        self.logger.info("--------------------")
        return


if __name__ == "__main__":
    # 测试示例
    from config import Config
    from model import SiameseNetwork
    import logging

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # 初始化模型和评估器
    model = SiameseNetwork(Config)
    evaluator = Evaluator(Config, model, logger)
    # 模拟评估
    evaluator.eval(epoch=1)