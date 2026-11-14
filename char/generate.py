import torch
from dataset import CharsDataset
from model import GPT, GPTConfig
import json

device = 'cuda' if torch.cuda.is_available() else 'cpu'
dataset = CharsDataset("docs\\corpus.txt", "docs\\vocab.json", block_size=384)

config = GPTConfig(vocab_size=len(dataset.vocab), block_size=384, n_embd=512, n_layer=6, n_head=8)
dataset = CharsDataset("docs\\corpus.txt", "docs\\vocab.json", block_size=384)
model = GPT(config).to(device)
checkpoint = torch.load("gpt_model.pth", map_location=device)
model.load_state_dict(checkpoint)

model.eval()  # 设置模型为推理模式

start_prompt = input("请输入开头提示词：")
# 把开头提示词转换成数字ID序列
start_ids = [dataset.vocab.get(ch, dataset.vocab["<unk>"]) for ch in start_prompt]

# 提示词向量化
prompt_tensor = torch.tensor(start_ids,dtype=torch.long).unsqueeze(0).to(device)

generated_ids = model.generate(idx=prompt_tensor,max_length=150,temperature=0.7,top_k=5)

with open("docs/id_to_char.json", "r", encoding="utf-8") as f:
    id_to_char = json.load(f)
    # 因为 JSON 的键必须是字符串，所以读取出来后，要把键转回整数
    id_to_char = {int(k): v for k, v in id_to_char.items()}

generated_text = ''.join([id_to_char.get(i, "<unk>") for i in generated_ids[0].tolist()])

print(f"\n 生成的文本: \n{generated_text}")
