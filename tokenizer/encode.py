import json
import numpy as np
import os


# paths
train_text_path = "data/train.txt"
vocab_path = "tokenizer/vocab.json"
output_path = "data/train.npy"


# load vocab
with open(vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)


# read text
with open(train_text_path, "r", encoding="utf-8") as f:
    text = f.read()


# encode characters
encoded = np.array([vocab[c] for c in text], dtype=np.int32)


# save numpy file
os.makedirs("data", exist_ok=True)
np.save(output_path, encoded)


print("Encoding complete")
print("Total tokens:", len(encoded))
print("Saved to:", output_path)