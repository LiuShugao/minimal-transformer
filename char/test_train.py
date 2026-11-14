'''
测试token是否成功向量化了
'''

import torch
from dataset import CharsDataset
from torch.utils.data import DataLoader
from model import GPT, GPTConfig
import json
import time

# 1. 加载数据集
# block_size=64表示每个样本的长度为64个字符，样本里的所有数据一次性加载到dataset里，dataset的长度就是总字数-64
dataset = CharsDataset("docs\\corpus.txt","docs\\vocab.json",block_size=384)
# 创建DataLoader，batch_size=4表示每次取4个样本，shuffle=True表示打乱顺序
dataloader = DataLoader(dataset, batch_size=32, shuffle=True,pin_memory=True, num_workers=0)

# 2.初始化模型
config = GPTConfig(vocab_size=len(dataset.vocab),block_size=384,n_embd=512,n_layer=6,n_head=8)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"正在使用设备: {device}")
model = GPT(config).to(device)

# 新增：定义优化器（Adam）
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
# 新增：定义学习率调度器
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10000)

# 加载之前训练好的权重
model.load_state_dict(torch.load("gpt_model.pth"))
print("已加载历史权重！")

print("模型初始化完成，开始训练...")

model.train()  # 设置模型为训练模式

# 👇 梯度累积参数
gradient_accumulation_steps = 8  # 每次攒 8 个 batch (8 * 8 = 等效于 batch_size 64)
# 注意：如果是刚开始跑，先把 load_state_dict 注释掉！

for step, (x, y) in enumerate(dataloader):
    
    if step >= 20000:  # 测试20000个batch
        break
    # 把数据搬到 GPU
    x, y = x.to(device), y.to(device)

    torch.cuda.synchronize()
    start_time = time.time()

    #把x输入模型，得到logits和loss
    logits, loss = model(x, y)
    loss = loss / gradient_accumulation_steps  # 梯度累积
    loss.backward()        # 反向传播计算梯度
    if (step + 1) % gradient_accumulation_steps == 0:
        optimizer.step()       # 更新参数
        optimizer.zero_grad(set_to_none=True)  # 清空梯度
        # 在循环里，每次 optimizer.step() 之后加上：
        scheduler.step()

    if step % 50 == 0:
        torch.cuda.synchronize()
        end_time = time.time()
        elapsed = end_time - start_time
        # 这里的 loss.item() 要乘回来才是真实的
        print(f"第 {step} 步, Loss: {loss.item() * gradient_accumulation_steps:.4f} | 耗时: {elapsed:.2f} 秒")

# ================== 测试生成 ===========================
print("\n---训练完成，开始测试生成---\n")

model.eval()  # 设置模型为推理模式

# 给模型一个开头
start_prompt = "阳光洒在金色的沙滩上，小明和他的朋友"
# 把开头提示词转换成数字ID序列
start_ids = [dataset.vocab.get(ch, dataset.vocab["<unk>"]) for ch in start_prompt]

# 变成模型需要的[1, T]形状的张量
input_ids = torch.tensor([start_ids], dtype=torch.long).to(device)

# 让模型生成后续的字符
generated_ids = model.generate(input_ids, max_length=300, temperature=0.8, top_k=20)

with open("docs/id_to_char.json", "r", encoding="utf-8") as f:
    # 因为 JSON 的键必须是字符串，所以读取出来后，要把键转回整数
    id_to_char_raw = json.load(f)
    id_to_char = {int(k): v for k, v in id_to_char_raw.items()}

# 把生成的ID序列转换回字符
generated_text = ''.join([id_to_char.get(i, "<unk>") for i in generated_ids[0].tolist()])

print(f"\n 生成的文本: \n{generated_text}")

# 保存模型权重
torch.save(model.state_dict(), "gpt_model.pth")
print("模型已保存为 gpt_model.pth")


