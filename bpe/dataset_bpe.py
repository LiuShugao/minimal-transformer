import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import json

class BPEDataset(Dataset):
    def __init__(self, corpus_path, tokenizer_path, block_size=384):
        self.block_size = block_size
        
        # 加载训练好的 BPE 分词器
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # 获取真实的词表大小（用于模型初始化）
        self.vocab_size = self.tokenizer.get_vocab_size()

        # 读取语料
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # 用 BPE 将整个文本编码成 ID 列表
        print("正在将语料转换为 BPE Token ID...")
        encoded = self.tokenizer.encode(text)
        self.data_ids = encoded.ids
        
        # 转为张量
        self.tensor_data = torch.tensor(self.data_ids, dtype=torch.long)
        print(f"BPE 转换完成！总 Token 数量: {len(self.tensor_data)}")

    def __len__(self):
        return len(self.tensor_data) - self.block_size

    def __getitem__(self, idx):
        x = self.tensor_data[idx : idx + self.block_size]
        y = self.tensor_data[idx + 1 : idx + self.block_size + 1]
        return x, y