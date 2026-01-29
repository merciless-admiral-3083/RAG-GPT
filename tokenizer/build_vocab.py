import json
import os

# path to training text
train_text_path = "data/train.txt"
vocab_path = "tokenizer/vocab.json"

# read full text
with open(train_text_path, "r", encoding="utf-8") as f:
    text = f.read()

# build character-level vocabulary
chars = sorted(list(set(text)))
vocab = {ch: i for i, ch in enumerate(chars)}

# save vocab
os.makedirs("tokenizer", exist_ok=True)
with open(vocab_path, "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

print("Vocabulary built")
print("Vocab size:", len(vocab))