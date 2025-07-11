import sentencepiece as spm

trainjson1 = "/home/kevingenghaopeng/MDD/mpl-mdd/data/train_unlabeled_erj.json"
trainjson2 = "/home/kevingenghaopeng/MDD/mpl-mdd/data/train_unlabeled.json"


import json
from collections import Counter
import sentencepiece as spm
import os

# ========== 配置 ==========
json_files = [
"/home/kevingenghaopeng/MDD/mpl-mdd/data/train_unlabeled_erj.json",
"/home/kevingenghaopeng/MDD/mpl-mdd/data/train_unlabeled.json",
]

output_dir = "output_bpe_dict"
vocab_size = 1000
model_prefix = os.path.join(output_dir, "bpe_model")

os.makedirs(output_dir, exist_ok=True)

# ========== Step 1: 读取 JSON，提取 wrd 字段 ==========
all_sentences = []
for file_path in json_files:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for entry in data.values():
            if "wrd" in entry and isinstance(entry["wrd"], str):
                sentence = entry["wrd"].strip().upper()
                if sentence:
                    all_sentences.append(sentence)

# 写入纯文本语料（用于 BPE 训练）
corpus_path = os.path.join(output_dir, "corpus.txt")
with open(corpus_path, "w", encoding="utf-8") as f:
    f.write("\n".join(all_sentences))

# ========== Step 2: 生成 Word-level Dictionary ==========
word_counter = Counter()
for line in all_sentences:
    word_counter.update(line.split())

word_dict_path = os.path.join(output_dir, "word_dictionary.txt")
with open(word_dict_path, "w", encoding="utf-8") as f:
    for word, count in word_counter.most_common():
        f.write(f"{word} {count}\n")

print(f"[✔] Word dictionary saved to: {word_dict_path}")

# ========== Step 3: 训练 BPE 模型 ==========
spm.SentencePieceTrainer.Train(
    input=corpus_path,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type="bpe",
    character_coverage=1.0
)

print(f"[✔] BPE model saved to: {model_prefix}.model")
print(f"[✔] BPE vocab saved to: {model_prefix}.vocab")