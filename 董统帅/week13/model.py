from config import Config
from transformers import AutoConfig, AutoModelForTokenClassification
from torch.optim import Adam, SGD

config = AutoConfig.from_pretrained(Config["pretrain_model_path"])
config.num_hidden_layers = Config['num_layers']
config.num_labels = 9
TorchModel = AutoModelForTokenClassification.from_pretrained(Config["pretrain_model_path"], config=config)

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)
