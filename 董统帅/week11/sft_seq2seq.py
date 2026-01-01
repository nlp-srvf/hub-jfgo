"""
大语言模型SFT（Supervised Fine-Tuning）微调示例
使用trl框架进行监督式微调
"""

import torch
# from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)
from trl import SFTTrainer, SFTConfig

# 配置
# 使用本地模型路径，确保可以离线运行
# 首次运行前，请先执行 python download_model.py 下载模型
MODEL_NAME = "./models/Qwen2-0.5B-Instruct"  # 本地模型路径
OUTPUT_DIR = "./output"
MAX_SEQ_LENGTH = 512

# 构造中文问答数据
def create_chinese_dataset(tokenizer):
    """构造简单的中文问答数据集"""
    data = [
        {
            "instruction": "请介绍一下人工智能。",
            "input": "",
            "output": "人工智能（AI）是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。这包括学习、推理、感知和自然语言处理等能力。"
        },
        {
            "instruction": "什么是深度学习？",
            "input": "",
            "output": "深度学习是机器学习的一个子领域，使用具有多个层（深度）的神经网络来学习数据的复杂模式和表示。它模仿人脑的神经网络结构。"
        },
        {
            "instruction": "Python中的列表和元组有什么区别？",
            "input": "",
            "output": "列表（list）是可变的，可以修改、添加或删除元素，使用方括号[]。元组（tuple）是不可变的，一旦创建就不能修改，使用圆括号()。"
        },
        {
            "instruction": "解释一下什么是监督学习。",
            "input": "",
            "output": "监督学习是机器学习的一种方法，使用标记的训练数据来训练模型。模型学习输入和输出之间的映射关系，然后可以对新的未标记数据进行预测。"
        },
        {
            "instruction": "如何提高模型的泛化能力？",
            "input": "",
            "output": "提高模型泛化能力的方法包括：1) 增加训练数据量和多样性 2) 使用正则化技术（如Dropout、L2正则化）3) 数据增强 4) 交叉验证 5) 防止过拟合。"
        },
        {
            "instruction": "什么是Transformer模型？",
            "input": "",
            "output": "Transformer是一种基于注意力机制的神经网络架构，由编码器和解码器组成。它摒弃了RNN和CNN，完全依赖注意力机制来处理序列数据，成为现代NLP的基础架构。"
        },
        {
            "instruction": "请解释一下梯度下降算法。",
            "input": "",
            "output": "梯度下降是一种优化算法，用于最小化损失函数。它通过计算损失函数对参数的梯度，然后沿着梯度反方向更新参数，逐步接近最优解。学习率控制每次更新的步长。"
        },
        {
            "instruction": "什么是迁移学习？",
            "input": "",
            "output": "迁移学习是将在一个任务或领域上学到的知识应用到另一个相关任务上的技术。它允许模型利用预训练的知识，从而在目标任务上更快地学习和获得更好的性能。"
        },
        {
            "instruction": "如何处理自然语言处理中的文本分类问题？",
            "input": "",
            "output": "文本分类的常见步骤包括：1) 文本预处理（分词、去停用词）2) 特征提取（词袋、TF-IDF、词向量）3) 选择分类算法（朴素贝叶斯、SVM、神经网络）4) 训练和评估模型。"
        },
        {
            "instruction": "请介绍一下大语言模型。",
            "input": "",
            "output": "大语言模型（LLM）是拥有数十亿甚至千亿参数的深度学习模型，通过在海量文本数据上预训练获得语言理解能力。它们可以执行各种NLP任务，如文本生成、问答、翻译等。"
        }
    ]
    
    # 将数据转换为对话格式（Qwen格式）
    max_length = 256
    reserved_length = 20
    formatted_data = []
    for item in data:
        # 使用Qwen的对话格式
        prompt = f"{item['output']}<|im_end|>"
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        text = tokenizer(text, max_length=max_length-reserved_length, truncation=True, return_tensors='pt')
        text_len = len(text['input_ids'][0])
        if text_len == max_length - reserved_length:
            continue
        remaining_length = max_length - text_len
        label = tokenizer(prompt, max_length=remaining_length, truncation=True, return_tensors='pt')
        input_ids = torch.cat([text['input_ids'], label['input_ids']], dim=1)
        attention_mask = torch.cat([text['attention_mask'], label['attention_mask']], dim=1)
        padding_len = max_length - len(input_ids[0])
        if padding_len > 0:
            padding_ids = torch.full((1, padding_len), tokenizer.pad_token_id, dtype=torch.long)
            attn_padding_ids = torch.full((1, padding_len), 0, dtype=torch.long)
            input_ids = torch.cat([input_ids, padding_ids], dim=1)
            attention_mask = torch.cat([attention_mask, attn_padding_ids], dim=1)
        input_ids = input_ids[:, :max_length]
        attention_mask = attention_mask[:, :max_length]
        label = input_ids.clone()
        label[:, :text_len] = -100
        pad_mask = (input_ids == tokenizer.pad_token_id)
        label[pad_mask] = -100
        formatted_data.append({"input_ids": input_ids.squeeze(0), "attention_mask": attention_mask.squeeze(0), "labels": label.squeeze(0)})
    
    return formatted_data

def load_model_and_tokenizer(model_name):
    """加载模型和分词器"""
    print(f"正在加载模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 使用float32以避免fp16训练时的梯度缩放问题
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        dtype=torch.float32,  # 使用float32以确保训练稳定性
        device_map="auto" if torch.cuda.is_available() else None
    )
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def main():
    """主训练函数"""
    print("=" * 50)
    print("开始SFT微调训练")
    print("=" * 50)
    
    # 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer(MODEL_NAME)


    # 创建数据集
    print("\n正在构造训练数据...")
    train_data = create_chinese_dataset(tokenizer)
    dataloader = DataLoader(train_data, batch_size=4, shuffle=True)
    print(f"训练样本数量: {len(train_data)}")

    epoch = 10
    learning_rate = 1e-5
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    model.train()
    for epoch_index in range(epoch):
        total_loss = 0
        for sample_data in dataloader:
            optimizer.zero_grad()
            output = model(input_ids=sample_data['input_ids'],
                  attention_mask=sample_data['attention_mask'],
                  labels=sample_data['labels'])
            loss = output.loss
            total_loss += loss.detach().item()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        print('average loss is: ', avg_loss)


    
    # 保存模型
    print(f"\n训练完成，正在保存模型到 {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("=" * 50)
    print("训练完成！")
    print("=" * 50)

if __name__ == "__main__":
    main()

    # model, tokenizer = load_model_and_tokenizer(OUTPUT_DIR)
    # x = "描述一下人工智能"

    # 分词并提取 input_ids
    # inputs = tokenizer(x, return_tensors="pt")
    # input_ids = inputs.input_ids  # 提取 input_ids 张量
    #
    # # 生成输出
    # with torch.no_grad():
    #     outputs = model.generate(input_ids, max_new_tokens=512)
    #     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # print(response)
    # print(outputs)
