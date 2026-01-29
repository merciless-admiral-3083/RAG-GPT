import os
import json
import torch

print("="*60)
print("🔍 GPT Project Diagnostics")
print("="*60)

# Check data files
print("\n📁 Data Files:")
files_to_check = [
    ("data/text_corpus.txt", "Training corpus"),
    ("data/train.npy", "Tokenized training data"),
    ("data/instruction_clean.json", "Instruction dataset"),
]

for filepath, description in files_to_check:
    if os.path.exists(filepath):
        size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        print(f"  ✅ {description}: {filepath} ({size:.1f} MB)")
    else:
        print(f"  ❌ {description}: {filepath} (NOT FOUND)")

# Check instruction data quality
print("\n📊 Instruction Data Quality:")
if os.path.exists("data/instruction_clean.json"):
    with open("data/instruction_clean.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Total samples: {len(data)}")
    print(f"  First 3 samples:")
    for i, sample in enumerate(data[:3]):
        print(f"\n  Sample {i+1}:")
        print(f"    Q: {sample['instruction'][:60]}...")
        print(f"    A: {sample['output'][:60]}...")
    
    # Check for mismatches
    mismatches = 0
    for sample in data[:50]:  # Check first 50
        q = sample['instruction'].lower()
        a = sample['output'].lower()
        # Simple heuristic: if question is about X, answer should mention X
        if "recursion" in q and "recursion" not in a:
            mismatches += 1
        if "stack" in q and "stack" not in a and "queue" in a:
            mismatches += 1
        if "queue" in q and "queue" not in a and "stack" in a:
            mismatches += 1
    
    if mismatches > 5:
        print(f"\n  ⚠️  WARNING: Found {mismatches} potential Q/A mismatches in first 50 samples!")
        print(f"     Consider regenerating with: python generate_instructions_fixed.py")
    else:
        print(f"\n  ✅ Data quality looks good!")

# Check model checkpoints
print("\n🤖 Model Checkpoints:")
log_dirs = ["log/main", "log/final", "log/toy"]
found_models = []

for log_dir in log_dirs:
    if os.path.exists(log_dir):
        files = [f for f in os.listdir(log_dir) if f.endswith('.pt')]
        if files:
            print(f"\n  📂 {log_dir}:")
            for f in sorted(files)[-5:]:  # Show last 5
                filepath = os.path.join(log_dir, f)
                size = os.path.getsize(filepath) / (1024 * 1024)  # MB
                print(f"    - {f} ({size:.1f} MB)")
                found_models.append(filepath)

if not found_models:
    print("  ❌ No model checkpoints found!")
    print("     You need to train a model first with: python train.py")
else:
    print(f"\n  ✅ Found {len(found_models)} model checkpoint(s)")

# Check if chat model exists
print("\n💬 Chat Model:")
chat_model_path = "log/main/model_chat.pt"
if os.path.exists(chat_model_path):
    size = os.path.getsize(chat_model_path) / (1024 * 1024)
    print(f"  ✅ Chat model exists: {chat_model_path} ({size:.1f} MB)")
    
    # Try to load and inspect
    try:
        ckpt = torch.load(chat_model_path, map_location='cpu')
        config = ckpt['config']
        print(f"\n  Model Configuration:")
        print(f"    - Layers: {config['n_layer']}")
        print(f"    - Heads: {config['n_head']}")
        print(f"    - Embedding dim: {config['n_embd']}")
        print(f"    - Block size: {config['block_size']}")
        print(f"    - Vocab size: {config['vocab_size']}")
    except Exception as e:
        print(f"  ⚠️  Could not inspect model: {e}")
else:
    print(f"  ❌ Chat model not found at: {chat_model_path}")
    print("     Create it with: python instruction_train.py")

# Recommendations
print("\n" + "="*60)
print("📋 Recommendations:")
print("="*60)

if not found_models:
    print("\n1️ TRAIN BASE MODEL:")
    print("   python train.py --max_steps 2000 --device cpu")
    print("   (This will take 30-60 minutes)")

elif not os.path.exists(chat_model_path):
    print("\n1️ CREATE CHAT MODEL:")
    latest_base = max(found_models, key=os.path.getmtime)
    print(f"   python instruction_train.py --base_model {latest_base} --out_dir log/main --device cpu")
    print("   (This will take 5-10 minutes)")

else:
    print("\nEverything looks ready!")
    print("\nSTART CHATTING:")
    print("   python chat_improved.py --model log/main/model_chat.pt --device cpu")
    print("\n   Or use original:")
    print("   python chat.py --model log/main/model_chat.pt --device cpu")

# Check for data quality issues
if os.path.exists("data/instruction_clean.json"):
    with open("data/instruction_clean.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Check first few for obvious mismatches
    issues = []
    for i, sample in enumerate(data[:10]):
        q = sample['instruction'].lower()
        a = sample['output'].lower()
        
        if "time complexity" in q and "space complexity" in a and "time" not in a:
            issues.append(f"Sample {i}: Q about time complexity, A about space complexity")
        if "recursion" in q and "stack" in a and "recursion" not in a:
            issues.append(f"Sample {i}: Q about recursion, A about stack")
    
    if issues:
        print("\nDATA QUALITY ISSUE:")
        for issue in issues[:5]:
            print(f"   - {issue}")
        print("\n   FIX THIS FIRST:")
        print("   python generate_instructions_fixed.py")
        print("   move instruction_clean_fixed.json data\\instruction_clean.json")
        print("   Then retrain the chat model!")

print("\n" + "="*60)
