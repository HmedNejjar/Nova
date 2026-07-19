# Nova

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-red)
![Status](https://img.shields.io/badge/status-in%20development-yellow)

A GPT-style language model built from scratch, with a focus on understanding and implementing the core components behind modern Large Language Models.

Nova is an ongoing exploration into how language models learn, reason, and generate text — from raw token prediction to instruction-following conversational systems.

## Overview

Nova aims to build a complete LLM pipeline, covering the fundamental stages involved in creating a conversational AI system:

- Data processing and tokenization
- Transformer-based language modeling
- Efficient autoregressive generation
- Continued model improvement through different training stages
- Instruction tuning for conversational abilities

The goal is not only to create a chatbot, but to understand the engineering and concepts behind the systems that power modern AI assistants.

## Current Features

- **Decoder-only Transformer architecture**, implemented from scratch in PyTorch
- **Custom Byte-Pair Encoding (BPE) tokenizer**, trained directly on a raw text corpus (no external tokenizer libraries)
- **Multi-head self-attention** with **Rotary Positional Embeddings (RoPE)**, computed from scratch
- **SwiGLU feed-forward network** and pre-norm residual blocks, in the style of modern LLaMA-esque decoders
- **Weight-tying** between the token embedding and output projection layers
- **KV-cache optimization** for efficient autoregressive inference, with correctly offset RoPE and causal masking during incremental decoding
- **Greedy autoregressive text generation**
- Dataset scaffolding for both **pretraining** (sliding-window next-token prediction) and **instruction tuning** (`<|user|>` / `<|assistant|>` formatted conversations with assistant-only loss masking)
- Mixed-precision training loop (AdamW, `torch.cuda.amp`) with per-epoch loss/accuracy tracking and Plotly-based training curves

## Project Structure

```
Nova/
├── GPT/
│   ├── Nova.py        # NovaLM: embedding, decoder, LM head, and generate()
│   ├── attention.py   # Multi-head self-attention with RoPE + KV cache
│   ├── decoder.py      # Decoder block (pre-norm attention + SwiGLU FFN)
│   ├── datasets.py     # VocabDataset (pretraining) & ChatBotDataset (instruction tuning)
│   └── train.py        # Training loop, evaluation, and metric plotting
├── Preprocess/
│   ├── tokenizer.py        # BPE tokenizer: training, encode/decode
│   ├── train_tokenizer.py  # Script to train and save a BPE vocab/merges
│   └── pos_embed.py        # Rotary Positional Embedding (RoPE) implementation
├── config.yaml         # Tokenizer, model, training, and dataset configuration
└── README.md
```

## Getting Started

### Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- PyYAML
- tqdm
- plotly

```bash
git clone https://github.com/HmedNejjar/Nova.git
cd Nova
pip install torch pyyaml tqdm plotly
```

### Configuration

All hyperparameters and paths are controlled through `config.yaml`:

```yaml
Tokenizer:
    type: "Byte-Pair Encoding"
    savepath: "Preprocess"
    corpus_path: "corpus.txt"
    vocab_size: 10_000

Model:
    embed_dim: 768
    num_heads: 8
    max_seq_len: 512
    num_layers: 6
    stride_coeff: 2
    learning_rate: 1e-4
    rope_base: 10_000

Train:
    epochs: 10
    batch_size: 8
```

### Training the tokenizer

The BPE tokenizer is trained directly on a raw text corpus (`Tokenizer.corpus_path` in `config.yaml`) and saves `vocab.json` / `merges.json` to the configured `savepath`.

```bash
cd Preprocess
python train_tokenizer.py
```

> Note: `train_tokenizer.py` and the `__main__` block in `tokenizer.py` currently point to a local absolute path — update these to match your own environment before running.

### Training the model

Once the tokenizer is trained and a tokenized dataset (pickled token ID lists, referenced under `Datasets` in `config.yaml`) is available, run:

```bash
python GPT/train.py
```

This trains `NovaLM` with AdamW and mixed precision, tracks train/eval loss and token-level accuracy per epoch, checkpoints the best model, and writes `loss_metrics.html` / `accuracy_metrics.html` (interactive Plotly charts) to the project root.

### Generating text

`NovaLM` exposes a `generate()` method that performs greedy decoding using the KV cache:

```python
from GPT.Nova import NovaLM
from Preprocess.tokenizer import BPE

tokenizer = BPE(vocab_size=10_000, savepath="Preprocess")
model = NovaLM(vocab_size=10_000, embed_dim=768, num_layers=6,
               num_heads=8, max_seq_len=512, rope_base=10_000)
# model.load_state_dict(torch.load("path/to/checkpoint.pt"))

output = model.generate(tokenizer, prompt="Once upon a time", max_new_tokens=100, device="cpu")
print(output)
```

## Training Approach

Nova follows a progressive learning approach:

1. **Language Learning** — The model first learns fundamental language patterns through large-scale text modeling. Currently trained on [SimpleStories](https://huggingface.co/datasets/SimpleStories/SimpleStories), a synthetic short-story corpus designed for training small, interpretable language models.
2. **Knowledge Expansion** — The model is further exposed to diverse and high-quality sources of information to improve its understanding of the world.
3. **Instruction Tuning** — The model is adapted toward conversational behavior using `<|user|>` / `<|assistant|>`-formatted dialogue with assistant-only loss masking, allowing it to better follow instructions and interact with users.

## Vision

The long-term goal of Nova is to develop a fully functional conversational language model while exploring the complete lifecycle of an LLM:

```
Raw Text
   ↓
Tokenization
   ↓
Pretraining
   ↓
Knowledge Expansion
   ↓
Instruction Tuning
   ↓
Conversational AI
```

## Why Nova?

Many AI systems are used without understanding what happens underneath. Nova is an attempt to bridge that gap by building the components from the ground up and exploring the ideas behind modern language models through implementation.

## Status

🚧 Nova is actively under development.

Implemented so far: BPE tokenizer training, a RoPE + SwiGLU decoder-only Transformer, KV-cached generation, and a pretraining loop with logged metrics. Instruction-tuning data handling (`ChatBotDataset`) is in place; a full instruction-tuning run and evaluation are next.

More components, experiments, and improvements will be added as the project evolves.
