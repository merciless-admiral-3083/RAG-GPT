import json
import os
import torch
import argparse
import tiktoken
from train import GPT, GPTConfig


def build_batch(enc, instruction, output, block_size):
    prompt = f"""### Instruction:
{instruction}

### Response:
"""

    full_text = prompt + output
    tokens = enc.encode(full_text)

    if len(tokens) < 10:
        return None, None

    tokens = tokens[: block_size + 1]

    x = torch.tensor(tokens[:-1], dtype=torch.long)
    y = torch.tensor(tokens[1:], dtype=torch.long)

    y_masked = torch.full_like(y, -100)

    prompt_tokens = enc.encode(prompt)
    start = len(prompt_tokens) - 1

    if start < len(y_masked):
        y_masked[start:] = y[start:]
    else:
        return None, None

    return x.unsqueeze(0), y_masked.unsqueeze(0)



def instruction_finetune(args):
    device = torch.device(args.device)
    enc = tiktoken.get_encoding("gpt2")

    # load base model
    ckpt = torch.load(args.base_model, map_location=device)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # load clean instruction data (JSON)
    with open("data/instruction_clean.json", "r", encoding="utf-8") as f:
        samples = json.load(f)

    print(f"Loaded {len(samples)} instruction samples")

    for epoch in range(3):
        total_loss = 0.0
        steps = 0

        for s in samples:
            instruction = s["instruction"].strip()
            output = s["output"].strip()

            x, y = build_batch(enc, instruction, output, config.block_size)
            if x is None:
                continue

            x, y = x.to(device), y.to(device)

            logits, loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            steps += 1

        print(f"Epoch {epoch+1} | loss {total_loss / steps:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": vars(model.config),
        },
        os.path.join(args.out_dir, "model_chat.pt")
    )

    print(f"Saved instruction-tuned model to {args.out_dir}/model_chat.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    instruction_finetune(args)