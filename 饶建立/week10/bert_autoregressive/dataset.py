
# dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from config import Config

class TextDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.load_data(file_path)

    def load_data(self, file_path):
        lines = []
        # 修改点：将 encoding='utf-8' 改为 'gbk' 或 'gb18030'
        # Windows下的中文文本通常是 gbk 编码
        try:
            with open(file_path, 'r', encoding='gbk') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 1:
                        lines.append(line)
        except UnicodeDecodeError:
            # 如果 gbk 也不行，尝试 utf-8（作为备选）
            print("GBK decode failed, trying UTF-8...")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if len(line) > 1:
                        lines.append(line)

        print(f"Loaded {len(lines)} lines from {file_path}")
        return lines

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]

        # 编码文本
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True, # 添加 [CLS], [SEP]
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        # 对于自回归任务，Labels 通常就是 Input IDs
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': input_ids.clone()
        }

def get_dataloader(config):
    tokenizer = BertTokenizer.from_pretrained(config.bert_path)
    dataset = TextDataset(config.train_path, tokenizer, config.max_len)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    return dataloader, tokenizer