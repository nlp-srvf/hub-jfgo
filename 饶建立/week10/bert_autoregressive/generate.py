
# generate.py
import torch
from transformers import BertTokenizer, BertLMHeadModel
from config import Config

def generate_text(start_text, max_length=50):
    cfg = Config()

    # 1. 加载 Tokenizer
    tokenizer = BertTokenizer.from_pretrained(cfg.bert_path)

    # 2. 加载模型结构
    model = BertLMHeadModel.from_pretrained(cfg.bert_path, is_decoder=True)

    # 3. 加载训练好的权重
    # map_location='cpu' 确保在只有CPU的机器上也能运行
    try:
        model.load_state_dict(torch.load(cfg.model_save_path, map_location=cfg.device))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Model file not found. Please train first.")
        return

    model.to(cfg.device)
    model.eval()

    # 4. 预处理输入文本
    input_ids = tokenizer.encode(start_text, add_special_tokens=False, return_tensors='pt').to(cfg.device)

    # 添加 [CLS] token (如果 bert-base-chinese 需要的话，通常生成任务最好有一个起始符)
    cls_token = torch.tensor([[tokenizer.cls_token_id]]).to(cfg.device)
    input_ids = torch.cat([cls_token, input_ids], dim=1)

    print(f"Input: {start_text}")
    print("Generating...", end="")

    # 5. 自回归生成循环
    with torch.no_grad():
        for _ in range(max_length):
            # 前向传播
            outputs = model(input_ids=input_ids)
            predictions = outputs.logits

            # 获取最后一个时间步的预测结果 (batch_size, seq_len, vocab_size) -> (vocab_size)
            last_token_logits = predictions[0, -1, :]

            # 简单贪婪搜索：取概率最大的词 (也可以改成 top-k 或 top-p 采样)
            predicted_id = torch.argmax(last_token_logits).item()

            # 遇到 [SEP] 停止生成
            if predicted_id == tokenizer.sep_token_id:
                break

            # 将预测的 token id 拼接到输入中，用于下一次预测
            input_ids = torch.cat([input_ids, torch.tensor([[predicted_id]]).to(cfg.device)], dim=1)

    # 6. 解码生成结果
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    # 去掉空格（中文bert tokenizer可能会产生空格）
    generated_text = generated_text.replace(" ", "")

    print(f"\n\nResult: {generated_text}")

if __name__ == "__main__":
    # 你可以在这里修改开头的文本
    start_input = "今天天气"
    generate_text(start_input)