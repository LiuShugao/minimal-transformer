import json

# 1. 收集字符集
chars = set()
# 加上 r 前缀，防止 \c 被当成转义字符
with open(r"docs\corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        chars.update(line)

# 2. 构建正向词表（注意这里循环变量换成了 ch）
vocab = {ch: i for i, ch in enumerate(sorted(chars))}

# 3. 加入特殊字符
for sp in ["<pad>", "<unk>", "<bos>", "<eos>"]:
    vocab[sp] = len(vocab)

# 4. 构建反向词表
id_to_char = {i: ch for ch, i in vocab.items()}

# 5. 保存为 JSON
with open(r"docs\vocab.json", "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

with open(r"docs\id_to_char.json", "w", encoding="utf-8") as f:
    json.dump(id_to_char, f, ensure_ascii=False, indent=2)

print(f"✅ 词表构建完成！总字符数: {len(vocab)}")