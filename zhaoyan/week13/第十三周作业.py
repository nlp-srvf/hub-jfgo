主要修改main文件与Config文件



训练结果
2026-01-22 17:26:33,717 - __main__ - INFO - 开始测试第20轮模型效果：
2026-01-22 17:27:02,399 - __main__ - INFO - PERSON类实体，准确率：0.794444, 召回率: 0.740933, F1: 0.766751
2026-01-22 17:27:02,399 - __main__ - INFO - LOCATION类实体，准确率：0.688679, 召回率: 0.618644, F1: 0.651781
2026-01-22 17:27:02,399 - __main__ - INFO - TIME类实体，准确率：0.519126, 召回率: 0.536723, F1: 0.527773
2026-01-22 17:27:02,399 - __main__ - INFO - ORGANIZATION类实体，准确率：0.628571, 召回率: 0.694737, F1: 0.659995
2026-01-22 17:27:02,400 - __main__ - INFO - Macro-F1: 0.651575
2026-01-22 17:27:02,400 - __main__ - INFO - Micro-F1 0.651697
2026-01-22 17:27:02,400 - __main__ - INFO - --------------------
2026-01-22 17:27:02,995 - __main__ - INFO - LoRA模型权重已保存到: model_output\epoch_20.pth (只保存可训练参数)

Config文件增加
# LoRA相关配置
"use_lora": True,  # 是否启用LoRA微调
"lora_r": 8,       # LoRA秩
"lora_alpha": 32,  # LoRA缩放系数
"lora_dropout": 0.1,  # LoRA dropout


main文件修改后
# -*- coding: utf-8 -*-
import torch
import os
import random
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
from peft import get_peft_model, LoraConfig

logging.basicConfig(level = logging.INFO,format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""

def main(config):
#创建保存模型的目录
if not os.path.isdir(config["model_path"]):
os.mkdir(config["model_path"])
# --------------------------------------------------------
def apply_lora_to_model(model, config):
"""
将LoRA微调应用到模型的BERT部分
"""
# 定义LoRA配置
lora_config = LoraConfig(
r=config.get("lora_r", 8), # LoRA秩
lora_alpha=config.get("lora_alpha", 32),
lora_dropout=config.get("lora_dropout", 0.1),
target_modules=["query", "key", "value"], # 对BERT的Q,K,V层应用LoRA
modules_to_save=["classify"], # 确保分类层不被冻结
)

# 应用LoRA到整个模型
model = get_peft_model(model, lora_config)

# 手动确保CRF层的参数可训练（因为LoRA可能会冻结它）
if hasattr(model, 'crf_layer'):
for param in model.crf_layer.parameters():
param.requires_grad = True

# 确保分类层也被训练（modules_to_save已经处理，这里双重保险）
for param in model.classify.parameters():
param.requires_grad = True

return model
# --------------------------------------------------------




#加载训练数据
train_data = load_data(config["train_data_path"], config)
#加载模型
model = TorchModel(config)
# --------------------------------------------------------
if config.get("use_lora", False):
logger.info("启用LoRA微调")
model = apply_lora_to_model(model, config)

# 打印可训练参数信息
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"LoRA可训练参数: {trainable_params}/{total_params} ({trainable_params / total_params * 100:.2f}%)")
# --------------------------------------------------------

# 标识是否使用gpu
cuda_flag = torch.cuda.is_available()
if cuda_flag:
logger.info("gpu可以使用，迁移模型至gpu")
model = model.cuda()
#加载优化器
optimizer = choose_optimizer(config, model)
#加载效果测试类
evaluator = Evaluator(config, model, logger)
#训练
for epoch in range(config["epoch"]):
epoch += 1
model.train()
logger.info("epoch %d begin" % epoch)
train_loss = []
for index, batch_data in enumerate(train_data):
optimizer.zero_grad()
if cuda_flag:
batch_data = [d.cuda() for d in batch_data]
input_id, labels = batch_data #输入变化时这里需要修改，比如多输入，多输出的情况
loss = model(input_id, labels)
loss.backward()
optimizer.step()
train_loss.append(loss.item())
if index % int(len(train_data) / 2) == 0:
logger.info("batch loss %f" % loss)
logger.info("epoch average loss: %f" % np.mean(train_loss))
evaluator.eval(epoch)
model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
# torch.save(model.state_dict(), model_path)
# --------------------------------------------------------
# 如果使用LoRA，只保存可训练参数（包括LoRA适配器和分类/CRF层）
if config.get("use_lora", False):
# 保存LoRA适配器和可训练层
saved_params = {
k: v.to("cpu")
for k, v in model.named_parameters()
if v.requires_grad
}
torch.save(saved_params, model_path)
logger.info(f"LoRA模型权重已保存到: {model_path} (只保存可训练参数)")
else:
# 保存完整模型
torch.save(model.state_dict(), model_path)
logger.info(f"完整模型权重已保存到: {model_path}")
# --------------------------------------------------------

return model, train_data

if __name__ == "__main__":
model, train_data = main(Config)
