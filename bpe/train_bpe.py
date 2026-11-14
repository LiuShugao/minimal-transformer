from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

#1.create a tokenizer with BPE model
tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))

#2.预处理：按字节切分
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

#3.解码器：按字节解码
tokenizer.decoder = decoders.ByteLevel()

#4.训练器：BPE算法，vocab_size=10000，min_frequency=2
trainer = trainers.BpeTrainer(vocab_size=10000, min_frequency=2, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"])

#5.训练分词器
tokenizer.train(["docs/corpus.txt"], trainer)
print("✅ BPE分词器训练完成！")

#6.保存分词器
tokenizer.save("docs/bpe_tokenizer.json")
print("✅ BPE分词器已保存至 docs/bpe_tokenizer.json ！")

#7.测试分词器
test_text = "阳光洒在金色的沙滩上，小明和他的朋友们在海边玩耍。"
encoded = tokenizer.encode(test_text)
print("\n--- 测试分词效果 ---")
print(f"原文: {test_text}")
print(f"切分出的 Token: {encoded.tokens}")
print(f"转换成的 ID: {encoded.ids}")