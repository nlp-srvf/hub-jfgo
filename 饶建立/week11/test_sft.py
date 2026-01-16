from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base_model_path = "./models/Qwen2-0.5B-Instruct"
lora_path = "./output/Qwen2-SFT-Checkpoints" # 训练输出目录

# 1. 加载基座
tokenizer = AutoTokenizer.from_pretrained(base_model_path)
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. 加载 LoRA 权重
model = PeftModel.from_pretrained(model, lora_path)

# 3. 推理
input_text = "写一首关于秋天的四行诗。"
messages = [
    {"role": "user", "content": input_text}
]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **inputs,
    max_new_tokens=512
)
print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))