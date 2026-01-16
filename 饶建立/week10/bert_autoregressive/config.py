
# config.py
import torch

class Config:
    def __init__(self):
        # 路径设置 (使用原始字符串 r"" 处理 Windows 路径)
        self.train_path = r"D:\Projects\nlp-study\饶建立\week10\corpus.txt"
        self.model_save_path = "./bert_gen_model.pth"

        # 预训练模型路径 (使用中文BERT作为基底)
        # 如果本地没有下载，Huggingface会自动下载
        self.bert_path = "bert-base-chinese"

        # 超参数
        self.max_len = 128       # 句子最大长度
        self.batch_size = 32
        self.epoch = 20
        self.learning_rate = 2e-5

        # 设备配置
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Device: {self.device}")