# GPT-RAG Assistant (Strictly context-grounded generation)

## 📋 Project Overview

GPT-RAG is a comprehensive framework for building a **Retrieval-Augmented Generation (RAG) system** that combines a custom-trained GPT language model with a semantic search mechanism to provide factual, Hallucination-resistant responses. The system is designed to answer questions using only information from a provided knowledge base, with built-in safeguards to refuse answering out-of-scope queries.

### Key Innovation
Rather than relying solely on the language model's training data (which can lead to hallucinations), this system:
1. **Retrieves** relevant context from a knowledge base using semantic similarity
2. **Gates** the response based on confidence thresholds
3. **Generates** answers exclusively from retrieved context
4. **Refuses** gracefully when knowledge is unavailable

---

## 🎯 Key Features

- **Custom GPT Model**: Transformer-based language model built from scratch using PyTorch
- **Semantic Search**: FAISS-indexed vector search using MiniLM embeddings for fast context retrieval
- **Confidence-based Gating**: Distance threshold mechanism to prevent answering without sufficient context
- **Instruction Fine-tuning**: Specialized training on instruction-response pairs for chat applications
- **Large-scale Pretraining**: Supports large-scale pretraining on FineWeb-Edu (10B tokens)
- **QA**: Context-grounded generative QA with strict retrieval gating
- **Safe Refusal**: Explicit "I don't know" responses for out-of-scope questions

---

## 🏗️ Architecture

```
┌─────────────────┐
│   User Query    │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│  1. Query Embedding             │
│     (Sentence Transformers)     │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  2. Semantic Search             │
│     (FAISS Index)               │
│     Retrieve top-k documents    │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  3. Confidence Gating           │
│     Distance < Threshold?       │
└────────┬─────────────────────────┤
         │ YES                 NO │
         ▼                        ▼
    ┌─────────┐         ┌──────────────┐
    │ Proceed │         │ Safe Refusal │
    └────┬────┘         └──────────────┘
         │
         ▼
┌─────────────────────────────────┐
│  4. Generate Answer             │
│     GPT Model with Context      │
└────────┬────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Final Answer   │
└─────────────────┘
```

---

## 📁 Project Structure

```
GPT_RAG/
├── train.py                    # Core GPT model implementation
├── chat.py                     # Interactive chat interface
├── instruction_train.py        # Fine-tuning on instruction datasets
├── fineweb.py                  # FineWeb-Edu dataset download & tokenization
├── test_rag.py                 # RAG system testing
├── diagnose.py                 # Diagnostic tools
├── generate_instructions_fixed.py  # Instruction dataset generation
│
├── rag/                        # RAG System Module
│   ├── config.py              # RAG configuration (thresholds, top-k, etc.)
│   ├── rag_retriever.py       # Semantic retrieval implementation
│   ├── build_rag_index.py     # FAISS index creation
│   └── convert_instruction_json.py  # Format conversion utilities
│
├── data/                       # Training Data
│   ├── instruction_clean.json  # Cleaned instruction-response pairs
│   ├── instructions.txt        # Raw instructions
│   ├── text_corpus.txt         # Text corpus for knowledge base
│   ├── train.npy               # Tokenized training data
│   ├── raw/                    # Raw data files
│   └── clean/                  # Cleaned data files
│
├── edu_fineweb10B/             # FineWeb-Edu Dataset (10B tokens)
│   ├── edufineweb_train_000001.npy  # Tokenized shards
│   ├── edufineweb_train_000002.npy
│   └── ... (57 total shards)
│
├── rag_data/                   # RAG-specific data
├── rag_index/                  # FAISS index & metadata
│   ├── index.faiss             # Vector index
│   └── data.json               # Document metadata
│
├── log/                        # Model checkpoints & logs
│   ├── config.json             # Training configuration
│   └── model_chat.pt           # Fine-tuned chat model
│
├── model_chat/                 # Chat model artifacts
├── tokenizer/                  # GPT-2 tokenizer files
│
└── README.md                   # This file
```

---

## 🛠️ Component Details

### 1. **train.py** - GPT Language Model
Implements a transformer-based GPT model from scratch:

**Architecture:**
- **Causal Self-Attention**: Multi-head attention with causal masking for autoregressive generation
- **MLP Layers**: Feed-forward networks with GELU activation
- **Block Stacking**: 4 transformer blocks (configurable)
- **Configuration**:
  - Block size: 128 tokens
  - Embedding dim (n_embd): 256
  - Attention heads (n_head): 4
  - Layers (n_layer): 4
  - Vocabulary: 50,257 (GPT-2 tokenizer)

**Key Classes:**
- `CausalSelfAttention`: Multi-head self-attention with causal masking
- `MLP`: Feed-forward layer (4x expansion)
- `Block`: Transformer block combining attention + MLP
- `GPTConfig`: Configuration dataclass
- `GPT`: Full model class with weight initialization, optimizer configuration, and pretrained loading

**Features:**
- Support for loading pretrained GPT-2 weights
- Gradient clipping and weight decay optimization
- Custom weight initialization scheme (NANOGPT_SCALE_INIT)

### 2. **chat.py** - Interactive Chat Interface
`chat.py` is the interactive entrypoint that wires together the FAISS-based RAG retriever and the local GPT model. It implements a hybrid strategy that
- attempts to answer from retrieved context first (RAG), and
- falls back to constrained GPT generation when RAG confidence is low.

**Highlights (from the current implementation):**
- Loads the fine-tuned chat model checkpoint (default: `log/main/model_chat.pt`) when available; otherwise runs in RAG-only mode.
- Uses `tiktoken` GPT-2 encoding for prompt/token handling.
- Applies a strict context validation and sentence-extraction pipeline to avoid low-quality context.
- Hybrid decision logic: computes a RAG confidence score and decides between the RAG answer and GPT-generated answer.

**Minimal prompt used by `chat.py`:**
```
Context: {context[:350]}

Q: {question}
A:
```

This implementation does not rely on an external `prompt_template` file or variable — the prompt is constructed inline as shown above.

**Command-line Arguments (as implemented):**
```
--model        Path to model checkpoint (default: log/main/model_chat.pt)
--device       Device to run on (default: cpu)
--temperature  Sampling temperature (default: 0.4)
--top_k        Top-k filtering parameter for sampling (default: 50)
--max_tokens   Maximum tokens to generate per query (default: 100)
--rag_weight   RAG confidence weight threshold (default: 0.90)
--debug        Enable debug prints
```

**Runtime behavior notes:**
- Retrieved documents are filtered by distance threshold and a context-quality check before joining into a single context.
- The GPT generator uses top-k sampling, token-level repetition checks, and early stopping heuristics to avoid loops and low-quality outputs.
- When updating docs or examples, prefer the compact prompt above rather than the older multi-line template.

### 3. **instruction_train.py** - Instruction Fine-tuning
Specialized training script for adapting the model to follow instructions:

**Process:**
1. Load base pretrained GPT model
2. Build instruction-response pairs from JSON dataset
3. Mask loss computation to only train on response tokens
4. Train for 3 epochs with learning rate 1e-5
5. Save instruction-tuned model

**Key Features:**
- **Selective Loss Masking**: Only compute loss on the response portion, not the instruction
- **Batch Building**: Pads sequences to block size and creates x, y, y_masked tensors
- **Gradient Clipping**: Prevents exploding gradients
- **Validation**: Skips malformed examples (< 10 tokens)

### 4. **fineweb.py** - Large-scale Dataset Processing
Downloads and tokenizes the FineWeb-Edu 10B token dataset:

**Dataset:**
- Source: HuggingFace FineWeb-Edu (sample-10BT split)
- Size: 10 billion tokens
- Tokenizer: GPT-2 encoder
- Shard Size: 100M tokens per shard (results in 100 shards)

**Features:**
- Multiprocessing tokenization for speed
- Memory-efficient shard writing
- End-of-text token separation between documents
- Uint16 token representation for storage efficiency

### 5. **rag/rag_retriever.py** - Semantic Search
FAISS-based vector retrieval system:

**Architecture:**
- **Embeddings**: Sentence Transformers (all-MiniLM-L6-v2)
- **Index**: FAISS for fast approximate nearest neighbor search
- **Metadata**: JSON file mapping index positions to documents

**Methods:**
- `retrieve(query, top_k)`: Return top-k most similar documents with similarity distances

### 6. **rag/config.py** - RAG Configuration
System parameters for controlling retrieval and gating:

```python
BEST_DISTANCE_THRESHOLD = 1.2  # Confidence threshold
TOP_K_RETRIEVAL = 3             # Number of documents to retrieve
MAX_CHUNKS_RETURNED = 2         # Maximum chunks in final context
```

---

## 🚀 Getting Started

### Prerequisites
```bash
pip install torch
pip install tiktoken
pip install transformers
pip install sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install numpy
pip install datasets  # For FineWeb download
pip install tqdm
```

### Step 1: Prepare Training Data

#### Option A: Use FineWeb-Edu (10B tokens)
```bash
python fineweb.py
```
This downloads and tokenizes the FineWeb-Edu dataset into `edu_fineweb10B/` directory.

#### Option B: Use Your Own Data
Prepare a text file and tokenize it using the tokenization utilities in `data/`.

### Step 2: Pretrain Base Model
```bash
python train.py \
    --data_dir=edu_fineweb10B \
    --out_dir=log/main \
    --epochs=1 \
    --batch_size=32 \
    --device=cuda
```

This trains a base GPT model on the FineWeb-Edu data.

### Step 3: Build RAG Index

First, prepare your knowledge base text in `data/text_corpus.txt`, then:
```bash
cd rag
python build_rag_index.py
```

This:
1. Chunks the text corpus
2. Generates embeddings using MiniLM
3. Creates a FAISS index
4. Saves metadata to JSON

### Step 4: Fine-tune for Instructions
```bash
python instruction_train.py \
    --base_model=log/main/model_final.pt \
    --out_dir=log/main \
    --device=cuda
```

This fine-tunes the base model on the instruction dataset for better chat performance.

### Step 5: Run Interactive Chat
```bash
python chat.py \
    --model=log/main/model_chat.pt \
    --device=cuda \
    --temperature=0.2
```

---

## 💬 Usage Examples

### Interactive Chat Mode

**Example 1: Factual Question (with Context)**
```
Q: What is recursion?
Retrieved Context: "Recursion is a programming technique where a function calls itself..."
A: Recursion is a programming technique where a function calls itself to solve a problem by breaking it down into smaller, similar subproblems.
```

**Example 2: Out-of-Scope Question (no Context)**
```
Q: Who invented electricity?
Retrieved Context: (distance > threshold, insufficient match)
A: I don't know based on the given context.
```

**Example 3: Partial Context**
```
Q: What are the main types of machine learning?
Retrieved Context: "Supervised learning uses labeled data. Unsupervised learning finds patterns..."
A: Based on the provided context, there are at least two main types: supervised learning (which uses labeled data) and unsupervised learning (which finds patterns without labels).
```

### Programmatic Usage

```python
import torch
from train import GPT, GPTConfig
from rag.rag_retriever import RAGRetriever
import tiktoken

# Load model
ckpt = torch.load("log/main/model_chat.pt", map_location="cpu")
config = GPTConfig(**ckpt["config"])
model = GPT(config)
model.load_state_dict(ckpt["model"])

# Load RAG retriever
rag = RAGRetriever("rag_index/index.faiss", "rag_index/data.json")

# Retrieve context
results = rag.retrieve("What is machine learning?", top_k=3)
context = "\n".join([r["text"] for r in results])

# Generate answer
question = "What is machine learning?"
prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
# Use model to generate response...
```

---

## 🔧 Configuration & Tuning

### Model Hyperparameters (train.py)
```python
class GPTConfig:
    block_size: int = 128        # Context window
    vocab_size: int = 50257      # GPT-2 tokenizer size
    n_layer: int = 4             # Number of transformer blocks
    n_head: int = 4              # Number of attention heads
    n_embd: int = 256            # Embedding dimension
```

**Tuning Tips:**
- Larger `n_embd` and `n_layer` → More powerful but slower
- Larger `block_size` → Better for long-range dependencies
- More heads → Better feature extraction but more computation

### RAG Parameters (rag/config.py)
```python
BEST_DISTANCE_THRESHOLD = 1.2   # Lower = stricter (require more confidence)
TOP_K_RETRIEVAL = 3             # More docs = richer context but slower
MAX_CHUNKS_RETURNED = 2         # Limit context length to prevent token overflow
```

**Tuning Tips:**
- Increase threshold to be more conservative (refuse more often)
- Increase top-k for comprehensive answers
- Decrease threshold to trust model more

### Generation Parameters (chat.py)
```
--temperature      0.1-0.4 for factual/deterministic outputs
--top_k            5-50 (higher = more sampling diversity; default in script: 50)
--max_tokens       Maximum tokens to generate (tune to control response length; default: 100)
--rag_weight       Confidence threshold to prefer RAG answer over GPT (0.85-0.95 recommended)
```

Note: The code no longer exposes a separate `repetition_penalty` CLI parameter; repetition and loop-prevention are enforced by internal heuristics in `chat.py`.

**Contacts / Maintainer**

- **Name:** Jaspreet Nahal
- **Email:** jaspreetnahal100@gmail.com
- **Website:** https://tinyurl.com/jaspreetnahal
                   
```

---

## 📊 Data Files

### Training Data

**instruction_clean.json**
```json
[
  {
    "instruction": "What is recursion?",
    "output": "Recursion is a programming technique where..."
  },
  ...
]
```

**text_corpus.txt**
Knowledge base content, one document per line or separated by special markers.

**train.npy**
Tokenized training data in numpy uint16 format.

### RAG Files

**rag_index/index.faiss**
FAISS binary index for fast similarity search.

**rag_index/data.json**
```json
[
  {
    "text": "Document text chunk...",
    "metadata": {...}
  },
  ...
]
```

---

## 🧪 Testing & Validation

### Test RAG Retrieval
```bash
python test_rag.py
```

Basic test to verify FAISS index and retriever are working correctly.

### Diagnostic Tools
```bash
python diagnose.py
```

Runs diagnostics on:
- Model architecture
- Tokenizer functionality
- FAISS index integrity
- RAG configuration

---

## 📈 Performance & Scalability

### Model Scaling
- **Current**: 4-layer, 256-dim transformer (small)
- **Medium**: 12-layer, 768-dim (similar to GPT-2)
- **Large**: 24+ layers, 1024+ dim (more powerful)

### Data Scaling
- Current: 10B tokens from FineWeb-Edu
- Can extend with additional datasets
- Shard-based training for memory efficiency

### Inference Speed
- CPU: ~0.5-1.0 sec per query (model + retrieval)
- GPU: ~0.1-0.2 sec per query
- FAISS retrieval: <10ms for most queries

---

## 🐛 Troubleshooting

### Out of Memory
- Reduce `batch_size` in training
- Reduce `block_size` in GPTConfig
- Use gradient accumulation

### Poor Answer Quality
- Increase `TOP_K_RETRIEVAL` to get more context
- Lower `BEST_DISTANCE_THRESHOLD` to be less strict
- Fine-tune longer or with more instruction data

### Retrieval Misses
- Check text_corpus.txt is properly formatted
- Rebuild FAISS index: `python rag/build_rag_index.py`
- Verify embeddings are correctly computed

### Model Not Loading
- Check checkpoint path exists
- Verify device (cpu/cuda) is available
- Ensure config matches model weights

---

## 📚 Key References & Papers

- **Transformers**: Vaswani et al., "Attention Is All You Need" (2017)
- **GPT Series**: Radford et al., "Language Models are Unsupervised Multitask Learners" (2019)
- **RAG**: Lewis et al., "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (2020)
- **FAISS**: Johnson et al., "Billion-scale similarity search with GPUs" (2017)
- **FineWeb**: HuggingFace FineWeb-Edu dataset

---

## 🔐 Safety Features

1. **Context Grounding**: All answers must be grounded in retrieved context
2. **Confidence Gating**: Distance threshold prevents weak matches
3. **Safe Refusal**: Explicit "I don't know" for out-of-scope queries
4. **Masked Loss Training**: Ensures model learns instruction-following
5. **Repetition Prevention**: Penalty mechanism discourages hallucinated repetitions

---

## 🚀 Future Enhancements

- [ ] Implement dense passage retrieval (DPR) for better embeddings
- [ ] Add multi-document fusion for complex queries
- [ ] Implement reranking stage for context quality
- [ ] Support for dynamic knowledge base updates
- [ ] Integration with vector databases (Pinecone, Weaviate)
- [ ] Web search integration for current events
- [ ] Multilingual support
- [ ] Streaming generation for real-time responses
- [ ] Caching system for common queries
- [ ] Metrics & evaluation harness (BLEU, ROUGE, F1)

---

## 📝 License & Attribution

This project combines custom implementations with community datasets:
- **Model**: Custom PyTorch implementation
- **Data**: FineWeb-Edu (HuggingFace)
- **Embeddings**: Sentence Transformers (BAAI)
- **Search**: FAISS (Meta)

---

## 🤝 Contributing

To extend this project:

1. **Add new datasets**: Update `data/` and `fineweb.py`
2. **Improve retrieval**: Modify `rag/rag_retriever.py`
3. **Enhance generation**: Tune parameters in `chat.py`
4. **Add features**: Create new modules in appropriate directories

---

## 📧 Contact & Support

For issues, questions, or suggestions:
- Check existing test files
- Review diagnostic output
- Consult troubleshooting section above
