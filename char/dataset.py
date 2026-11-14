import torch
from torch.utils.data import Dataset,DataLoader
import json

class CharsDataset(Dataset):
    def __init__(self,corpus_path,vocab_path,block_size=64):

        self.block_size=block_size

        with open(vocab_path,"r",encoding="utf-8") as f:
            self.vocab = json.load(f)

        with open(corpus_path,"r",encoding="utf-8") as f:
            text = f.read()

        unk_id = self.vocab["<unk>"]
        self.data_ids = [self.vocab.get(ch,unk_id) for ch in text]

        self.tensor_data = torch.tensor(self.data_ids,dtype=torch.long)

    def __len__(self):
        return len(self.tensor_data) - self.block_size
    
    def __getitem__(self,idx):
        x=self.tensor_data[idx : idx + self.block_size]
        y=self.tensor_data[idx + 1: idx + self.block_size + 1]
        return x,y

'''
if __name__ == "__main__":
    dataset = CharsDataset("docs\\corpus.txt","docs\\vocab.json",block_size=64)
    print(f"词表大小：{len(dataset.vocab)}")
    print(f"预料总字数：{len(dataset.tensor_data)}")
'''