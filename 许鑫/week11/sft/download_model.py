"""
下载模型到本地文件夹，用于离线训练
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

MODEL_NAME = "Qwen/Qwen2-0.5B-Instruct"
LOCAL_MODEL_DIR = "./models/Qwen2-0.5B-Instruct"

def download_model():
    """下载模型和分词器到本地"""
    print("=" * 50)
    print("开始下载模型到本地")
    print(f"模型: {MODEL_NAME}")
    print(f"保存位置: {LOCAL_MODEL_DIR}")
    print("=" * 50)
    
    # 创建目录
    os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
    
    print("\n正在下载分词器和模型（这可能需要几分钟，请耐心等待）...")
    print("如果下载较慢，请耐心等待，模型大小约1GB...")
    
    # 下载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )
    tokenizer.save_pretrained(LOCAL_MODEL_DIR)
    print("分词器下载完成")
    
    # 下载模型
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16  # 使用float16减少内存占用
    )
    model.save_pretrained(LOCAL_MODEL_DIR)
    print("模型下载完成")
    
    print("\n" + "=" * 50)
    print("模型下载完成！")
    print(f"模型已保存到: {LOCAL_MODEL_DIR}")
    print("现在可以离线运行训练脚本了")
    print("=" * 50)

if __name__ == "__main__":
    try:
        download_model()
    except Exception as e:
        print(f"\n下载过程中出现错误: {e}")
        print("请检查网络连接，确保可以访问 Hugging Face")
        raise

