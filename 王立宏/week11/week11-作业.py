# coding:utf8

'''
尝试实现sft式的seq2seq训练

本脚本实现了基于 Qwen2.5-0.5B-Instruct 模型的监督微调(SFT)训练流程。
使用 trl 库的 SFTTrainer 进行模型微调，训练客服对话数据集。

主要功能：
1. 构造中文客服对话数据集
2. 加载预训练模型和分词器
3. 使用 SFTTrainer 进行模型微调
4. 保存训练好的模型
'''
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,  # 自动加载分词器
    AutoModelForCausalLM,  # 自动加载因果语言模型
    TrainingArguments  # 训练参数配置
)
from trl import SFTTrainer, SFTConfig  # trl库的监督微调训练器和配置

# 模型路径：本地已下载的 Qwen2.5-0.5B-Instruct 模型
MODEL_NAME = "./models/Qwen2.5-0.5B-Instruct"
# 输出目录：训练完成的模型和检查点将保存到此路径
OUTPUT_DIR = "./output"
# 最大序列长度：输入文本的最大token数量，超过将被截断
MAX_SEQ_LENGTH = 512


def create_chinese_dataset():
    """
    构造中文客服对话数据集

    Returns:
        list: 包含格式化对话数据的列表，每个元素是包含"text"字段的字典

    数据格式说明：
    - instruction: 用户的问题/指令
    - input: 额外的输入信息（本例中为空）
    - output: 期望的模型回答
    """
    # 定义原始的对话数据，包含5组客服问答
    data = [
        {
            "instruction": "你们的客服工作时间是？",
            "input": "",
            "output": "我们的在线客服团队工作时间为周一至周日，每日上午9点至晚上21点（夏令时为上午8点至晚上20点）。您可以随时联系我们获取帮助。"
        },
        {
            "instruction": "配送时间是多久？",
            "input": "",
            "output": "亲爱的顾客，我们的标准配送时间为3-5个工作日内送达，但具体的时间会根据您所在地区的物流情况和包裹重量有所变动。详情请参考您订单页面的预计到达日期。"
        },
        {
            "instruction": "退货的时间限制是多久？",
            "input": "",
            "output": "我们接受的无理由退货时间为购买后7天内。如果您需要因商品质量问题进行退货，此时间从收到并确认问题后的次日开始计算。超过时间范围将无法受理。"
        },
        {
            "instruction": "是否有货?",
            "input": "",
            "output": "能下单就是在售有货的哦亲亲"
        },
        {
            "instruction": "能否开具发票？",
            "input": "",
            "output": "是的，我们可以为所有购买的商品提供正规发票服务。如果您需要发票，请在订单提交时留下相关信息或联系我们的客服进行后续操作。"
        },
    ]
    # 将数据转换为 Qwen 模型所需的对话格式
    formatted_data = []
    for item in data:
        # 使用 Qwen 的特殊标记格式化对话
        # <|im_start|> 和 <|im_end|> 是 Qwen 模型的对话边界标记
        text = f"<|im_start|>user\n{item['instruction']}<|im_end|>\n<|im_start|>assistant\n{item['output']}<|im_end|>"
        formatted_data.append({"text": text})
    return formatted_data


def load_model_and_tokenizer():
    """
    加载预训练模型和分词器

    Returns:
        tuple: (model, tokenizer) 模型和分词器的元组
    """
    print(f"正在加载模型: {MODEL_NAME}")
    # 加载分词器，trust_remote_code=True 允许执行模型自定义的代码
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    # 加载模型
    # torch_dtype=torch.float32: 使用32位浮点精度（在CPU上训练时使用）
    # device_map="auto": 自动将模型分配到可用的GPU设备（如果可用）
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )
    # 设置 pad_token（填充token）
    # 某些模型没有定义 pad_token，此时使用 eos_token（结束token）作为替代
    # 这对于批处理训练是必需的，因为不同长度的序列需要填充到相同长度
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def main():
    """
    主训练流程

    执行完整的监督微调训练流程：
    1. 加载模型和分词器
    2. 准备训练数据集
    3. 配置训练参数
    4. 执行训练
    5. 保存训练好的模型
    """
    print("=" * 50)
    print("开始SFT微调训练")
    print("=" * 50)
    # 步骤1: 加载模型和分词器
    model, tokenizer = load_model_and_tokenizer()
    # 步骤2: 构造训练数据
    print("\n正在构造训练数据...")
    train_data = create_chinese_dataset()
    # 将列表数据转换为 Hugging Face Dataset 对象
    dataset = Dataset.from_list(train_data)
    print(f"训练样本数量: {len(dataset)}")
    # 步骤3: 配置 SFT 训练参数
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,  # 输出目录，保存模型和检查点
        num_train_epochs=3,  # 训练轮数：遍历整个数据集3次
        per_device_train_batch_size=2,  # 每个设备的批处理大小（每个GPU/CPU每次处理2个样本）
        gradient_accumulation_steps=4,  # 梯度累积步数：每4步更新一次梯度，等效批大小=2*4=8
        learning_rate=2e-5,  # 学习率：控制参数更新步长，较小的学习率更稳定
        warmup_steps=10,  # 预热步数：前10步使用较小的学习率逐步增加到目标值
        logging_steps=1,  # 日志记录频率：每步记录一次训练信息
        save_steps=50,  # 检查点保存频率：每50步保存一次模型
        save_total_limit=2,  # 保留的检查点数量：最多保留2个最新的检查点
        fp16=False,  # 不使用16位浮点混合精度训练
        bf16=False,  # 不使用BFloat16精度训练
        remove_unused_columns=False,  # 保留数据集中的所有列（某些数据集可能需要额外字段）
        report_to=None,  # 不向外部工具（如wandb、tensorboard）报告训练指标
        dataloader_pin_memory=False,  # 不固定内存（在CPU训练时可以提高内存利用率）
        max_length=MAX_SEQ_LENGTH,  # 最大序列长度：512个token
        dataset_text_field="text",  # 数据集中文本内容的字段名
    )
    # 步骤4: 创建 SFT 训练器
    trainer = SFTTrainer(
        model=model,  # 要训练的模型
        args=sft_config,  # 训练配置
        train_dataset=dataset,  # 训练数据集
        processing_class=tokenizer,  # 分词器（用于数据处理）
    )
    # 步骤5: 开始训练
    print("\n开始训练...")
    trainer.train()
    # 步骤6: 保存训练好的模型
    print(f"\n训练完成，正在保存模型到 {OUTPUT_DIR}")
    trainer.save_model()  # 保存模型权重
    tokenizer.save_pretrained(OUTPUT_DIR)  # 保存分词器
    print("=" * 50)
    print("训练完成！")
    print("=" * 50)


# 程序入口：当脚本被直接执行时，调用 main 函数
# 如果被作为模块导入，则不会执行 main 函数
if __name__ == "__main__":
    main()
