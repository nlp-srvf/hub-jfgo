import os
import torch
# from torch.utils.data import Dataset
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

# ================= 配置参数 =================
# 这里的路径需与 download_model.py 中的 LOCAL_MODEL_DIR 保持一致
MODEL_PATH = "./models/Qwen2-0.5B-Instruct"
OUTPUT_DIR = "./output/Qwen2-SFT-Checkpoints"

# 训练参数
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 2e-4
BATCH_SIZE = 4
NUM_EPOCHS = 3
GRADIENT_ACCUMULATION_STEPS = 4

# ================= 1. 准备数据 =================
# 为了演示，直接构造一个内存中的数据集
# 在实际场景中，应该加载 json 或 jsonl 文件
def get_dummy_dataset():
    data = [
        {
            "instruction": "请把这句话翻译成英文：今天天气真好。",
            "input": "",
            "output": "The weather is really nice today."
        },
        {
            "instruction": "介绍一下Python是什么。",
            "input": "",
            "output": "Python是一种广泛使用的高级编程语言，以其清晰的语法和代码可读性而闻名。它支持多种编程范式，包括面向对象、指令式、函数式和过程式编程。"
        },
        {
            "instruction": "写一首关于秋天的四行诗。",
            "input": "",
            "output": "金风送爽叶飘黄，\n雁阵南飞过大江。\n稻谷飘香农事忙，\n秋光绚丽胜春光。"
        },
        # 实际训练时数据量应更多
    ] * 10  # 复制几次以便能跑起来
    return Dataset.from_list(data)

# 格式化函数：将数据转换为 Qwen2 的 Chat 模板格式
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": example['instruction'][i] + ("\n" + example['input'][i] if example['input'][i] else "")},
            {"role": "assistant", "content": example['output'][i]}
        ]
        # 使用 tokenizer 的 apply_chat_template 自动处理 <|im_start|> 等特殊token
        # 注意：这里不进行 tokenize，只转成字符串，交给 SFTTrainer 处理
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        output_texts.append(text)
    return output_texts

# ================= 2. 加载模型与分词器 =================
print(f"正在加载本地模型: {MODEL_PATH}")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    use_fast=False # 有时候Qwen用 fast tokenizer 会有兼容性问题，视版本而定
)

# 确保 pad_token_id 正确
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16, # 如果是 CPU 训练，请改回 torch.float32
    trust_remote_code=True,
    device_map="auto" # 自动分配 GPU/CPU
)

# 开启梯度检查点以节省显存 (可选)
model.gradient_checkpointing_enable()

# ================= 3. 配置 LoRA (PEFT) =================
# 0.5B 模型虽然可以直接全量微调，但使用 LoRA 更高效且防遗忘
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,              # LoRA 秩，越大参数越多
    lora_alpha=32,    # LoRA 缩放系数
    lora_dropout=0.1,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] # Qwen 的全线性层
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # 打印可训练参数量

# ================= 4. 配置 Seq2Seq Masking =================
# 关键步骤：我们希望模型只学习 assistant 的回答，不学习 user 的提问。
# Qwen2 的 Chat 模板通常包含 "<|im_start|>assistant\n"
# 我们利用 DataCollatorForCompletionOnlyLM 来自动 Mask 掉 prompt 部分
response_template = "<|im_start|>assistant\n"
# 注意：response_template 需要根据 tokenizer 实际编码结果微调，
# 有时候可能是 token id 列表。简单起见，这里先尝试字符串匹配。

collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

# ================= 5. 设置训练参数 =================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    logging_steps=1,
    num_train_epochs=NUM_EPOCHS,
    save_steps=50,
    save_total_limit=2,
    fp16=True, # 开启混合精度训练 (需要 GPU)
    # use_cpu=True, # 如果没有显卡，取消注释这行，并把 fp16=False, model 加载时的 float16 改为 float32
    report_to="none" # 不上传 wandb
)

# ================= 6. 开始训练 =================
dataset = get_dummy_dataset()

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    tokenizer=tokenizer,
    peft_config=peft_config,
    formatting_func=formatting_prompts_func, # 应用格式化
    data_collator=collator, # 使用 Masking 策略
    max_seq_length=MAX_SEQ_LENGTH,
)

print("\n开始训练...")
trainer.train()

print("\n训练完成，正在保存模型...")
# 保存 LoRA 权重
trainer.save_model(OUTPUT_DIR)
print(f"模型权重已保存至: {OUTPUT_DIR}")