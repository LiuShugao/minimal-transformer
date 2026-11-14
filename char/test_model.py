'''
测试token是否成功向量化了
'''

import torch
from dataset import CharsDataset
from torch.utils.data import DataLoader
from model import GPT, GPTConfig

# 1. 加载数据集
# block_size=64表示每个样本的长度为64个字符，样本里的所有数据一次性加载到dataset里，dataset的长度就是总字数-64
dataset = CharsDataset("docs\\corpus.txt","docs\\vocab.json",block_size=64)
# 创建DataLoader，batch_size=4表示每次取4个样本，shuffle=True表示打乱顺序
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# 2.初始化模型
config = GPTConfig(vocab_size=len(dataset.vocab),block_size=64,n_embd=128,n_layer=4,n_head=4)
model = GPT(config)

# 新增：定义优化器（Adam）
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 3.拿一个batch的数据，看看模型的输出
for x,y in dataloader:
    #看下形状
    print(f"x的形状：{x.shape},y的形状：{y.shape}")

    #把x输入模型，得到logits和loss
    logits, loss = model(x, y)

    print(f"logits的形状：{logits.shape}")
    print(f"当前loss：{loss.item():.4f}")
   
    # 4.反向传播和优化（更新参数）
    optimizer.zero_grad()  # 清空上一步残留的梯度
    loss.backward()        # 反向传播计算梯度
    optimizer.step()       # 更新参数
    
    print("参数更新完成")

    break