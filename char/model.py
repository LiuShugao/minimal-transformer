import torch
import torch.nn as nn
import math

class Head(nn.Module):
    def __init__(self, config, head_size):
        super().__init__()
        self.config = config
        self.head_size = head_size

        # 初始化Q,K,V的权重矩阵W_Q, W_K, W_V
        self.key = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.query = nn.Linear(config.n_embd, self.head_size, bias=False)
        self.value = nn.Linear(config.n_embd, self.head_size, bias=False)

        # 生成一个下三角矩阵，用于掩码，防止模型看到未来的信息
        # torch.tril 把右上角全部置为0，保留左下角
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))

    def forward(self, x):
        B,T,C = x.shape

        # 1.计算Q,K,V
        k = self.key(x)      # (B,T,head_size)
        q = self.query(x)    # (B,T,head_size)
        v = self.value(x)    # (B,T,head_size)

        # 2.计算注意力分数，核心公式：Attention(Q,K,V) = softmax(QK^T/sqrt(d_k))V

        # QK^T的形状是(B,T,T)，表示每个样本的每个位置对所有位置的注意力分数
        att = q @ k.transpose(-2,-1) 

        # 缩放（除以sqrt(d_k)，防止数值过大导致softmax输出梯度消失）
        att = att / math.sqrt(self.head_size)  # (B,T,T)


        # 3.掩码：把未来的信息置为负无穷大，防止模型看到未来的信息
        att = att.masked_fill(self.tril[:T,:T] == 0, float("-inf"))

        # 4.概率归一化：对每一行进行softmax，得到注意力权重
        att = torch.softmax(att, dim=-1)  # (B,T,T)

        # 5.加权求和：把V加权求和，得到最终的输出
        out = att @ v  # (B,T,head_size)

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.head_size = config.n_embd // config.n_head

        # 初始化多个注意力头，每个头都是一个Head实例
        self.heads = nn.ModuleList([Head(config,self.head_size) for _ in range(config.n_head)])

        # 线性变换，把多个头的输出拼接起来映射回原来的维度
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        # 让每个头独立计算注意力，然后把它们的输出拼接起来
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,head_size*n_head) = (B,T,n_embd)

        # 线性变换，把拼接后的向量映射回原来的维度
        out = self.proj(out)  # (B,T,n_embd)

        return out


# 3.前馈神经网络（FFN）
class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),  # 扩展维度
            nn.ReLU(),                                    # 激活函数
            nn.Linear(4 * config.n_embd, config.n_embd),  # 压缩回原来的维度
            nn.Dropout(0.2)                               # 防止过拟合
        )

    def forward(self, x):
        return self.net(x)


# 4.完整的Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = MultiHeadAttention(config)
        self.ffn = FeedForward(config)
        # 两个LayerNorm层，分别用于注意力层和前馈神经网络层
        self.ln1 = nn.LayerNorm(config.n_embd)  # LayerNorm层
        self.ln2 = nn.LayerNorm(config.n_embd)  # LayerNorm层

    def forward(self, x):
        # 残差连接 x = x + SubLayer(LayerNorm(x))
        x = x + self.attn(self.ln1(x))  # 注意力层的残差链接
        x = x + self.ffn(self.ln2(x))   # 前馈神经网络层的残差链接
        return x


class GPTConfig:
    '''存放模型的超参数'''
    def __init__(self,vocab_size=5704,block_size=256,n_embd=512,n_layer=6,n_head=8):
        self.vocab_size=vocab_size
        self.block_size=block_size
        self.n_embd=n_embd              #词向量维度(这里设置的512)
        self.n_layer=n_layer
        self.n_head=n_head

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 1. Token Embedding: 查词表，将ID映射为向量
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)

        # 2. Position Embedding: 在ID向量上添加一个位置编码
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)

        # # 3.注意力层
        # self.attn = MultiHeadAttention(config)

        # 3. Transformer层堆叠
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layer)])

        # 4. LayerNorm层：对Transformer的输出进行归一化
        self.ln_f = nn.LayerNorm(config.n_embd)

        # 5. 输出层：把向量映射回词表大小的logits
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

    # 3.前向传播，将输入的ID序列映射为向量
    def forward(self, idx, targets=None):
        # 获取idx的形状(B,T)，B是batch_size，T是序列长度。idx是一个二维张量，存放了每个样本的ID序列
        B,T = idx.shape

        # 1.获取Token的向量表示
        token_emb = self.token_embedding(idx)          # (B,T,n_embd)

        # 2.获取Position的向量表示
        # torch.arange(T,device=idx.device)生成一个长度为T的张量，存放了每个位置的索引(0,1,2,...,T-1)
        pos = torch.arange(T, dtype=torch.long, device=idx.device)        # (T,)
        pos_emb = self.position_embedding(pos)         # (T,n_embd)

        # 3.相加(广播机制)：将pos_emb的形状从(T,n_embd)广播到(B,T,n_embd)；也就是把每个样本的Token Embedding都加上相同的Position Embedding
        x = token_emb + pos_emb                        # (B,T,n_embd)

        # 经过N层TransformerBlock堆叠，每一层都包含一个多头注意力机制和一个前馈神经网络
        x = self.transformer_blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B,T,vocab_size)

        # 如果传入targets，说明是在训练，需要计算损失
        loss = None 

        if targets is not None:
            # 计算交叉熵损失
            # logits的形状是(B,T,vocab_size)，targets的形状是(B,T)
            # 需要把logits展平为(B*T,vocab_size)，targets展平为(B*T,)
            B,T,C = logits.shape
            # 把logits展平为(B*T,vocab_size)，targets展平为(B*T,)
            logits_flat = logits.view(B*T,C)
            targets_flat = targets.view(B*T)
            # 计算交叉熵损失
            loss = nn.functional.cross_entropy(logits_flat, targets_flat)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, idx, max_length, temperature=1.0, top_k=None):
        # idx: (B,T) 输入的ID序列，B是batch_size，T是序列长度
        for _ in range(max_length):
            # 1.截断输入序列，保证长度不超过block_size
            idx_cond = idx[:, -self.config.block_size:]  # (B,T_cond)

            logits, _ = self.forward(idx_cond)

            # 2.取最后一个时间步的logits，形状是(B,vocab_size)
            logits = logits[:, -1, :] # / temperature

            # 3.如果top_k不为None，则只保留前top_k个最大的logits，其余置为负无穷大
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')

            # 4.计算softmax，得到每个词的概率分布
            probs = torch.softmax(logits, dim=-1)  # (B,vocab_size)

            # 5.从概率分布中采样一个词的ID
            next_id = torch.multinomial(probs, num_samples=1)  # (B,1)

            # 6.把采样得到的ID拼接到输入序列的末尾，作为下一次生成的输入
            idx = torch.cat((idx, next_id), dim=1)  # (B,T+1)

        return idx

