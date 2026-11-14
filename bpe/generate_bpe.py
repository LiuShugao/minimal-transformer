import torch
from model_bpe import GPT, GPTConfig
from tokenizers import Tokenizer

# 加载BPE数据集
# dataset = BPEDataset(r"docs\corpus.txt","docs/bpe_tokenizer.json",block_size=384)
tokenizer = Tokenizer.from_file("docs/bpe_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()

# 初始化设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config = GPTConfig(vocab_size=vocab_size,block_size=384,n_embd=512,n_layer=6,n_head=8)

model = GPT(config).to(device)

# 加载模型权重
checkpoint = torch.load("gpt_model.pth",map_location=device)
model.load_state_dict(checkpoint)

model.eval()  # 设置模型为推理模式

start_prompt = input("请输入提示词：").strip()
if not start_prompt:
    raise ValueError("提示词不能为空")

start_ids = tokenizer.encode(start_prompt).ids
start_tensor = torch.tensor([start_ids], dtype=torch.long).to(device)

generate_ids = model.generate(start_tensor, max_length=300, temperature=0.8, top_k=12)

generate_text = tokenizer.decode(generate_ids[0].tolist())

print(f"\n 生成的文本: \n{generate_text}")

