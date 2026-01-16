from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# MODEL_PATH = "./models/Qwen2.5-0.5B-Instruct"
MODEL_PATH = "./output"

def test_model():
    print("正在加载模型...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code = True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code = True,
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map = "auto" if torch.cuda.is_available() else None
    )

    test_questions = [
        "你们的客服工作时间是？",
        "配送时间是多久？",
        "退货的时间限制是多久？"
    ]

    print("\n开始测试...")
    print("="*50)

    for question in test_questions:
        prompt = f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(prompt,return_tensors = "pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k,v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens = 200,
                temperature = 0.7,
                top_p = 0.9,
                do_sample = True,
                pad_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.pad_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens = False)

        if "<|im_start|>assistant\n" in response:
            answer = response.split("<|im_start|>assistant\n")[1].split("<|im_end|>")[0].strip()
        else:
            answer = response.replace(prompt, "").replace("<|im_end|>", "").strip()

        print(f"问题: {question}")
        print(f"回答: {answer}")
        print("-" * 50)

if __name__ == "__main__":
    test_model()