import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(1, str(ROOT))

import yaml
import torch
from GPT.Nova import NovaLM
from Preprocess.tokenizer import BPE

from safetensors.torch import save_model, load_model

with open(ROOT / Path(r"config.yaml"), 'r') as f:
    config = yaml.safe_load(f)
    
model_config = config["Model"]
tokenizer_config = config["Tokenizer"]

NUM_LAYERS = model_config["num_layers"]
EMBED_DIM = model_config["embed_dim"]
NUM_HEADS = model_config["num_heads"]
MAX_SEQ_LEN = model_config["max_seq_len"]
STRIDE_COEFF = model_config["stride_coeff"]
LR = float(model_config["learning_rate"])
DROPOUT = float(model_config["dropout"])
ROPE_BASE = model_config["rope_base"]
MODEL_SAVE_PATH = ROOT / Path(model_config["savepath"])

TOP_K = model_config["top_k"]
TEMP = model_config["temperature"]

VOCAB_SIZE = tokenizer_config["vocab_size"]
SAVEPATH = tokenizer_config["savepath"]

MAX_TOKENS = 80

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

bpe = BPE(vocab_size= VOCAB_SIZE, savepath= SAVEPATH)
print(f"{bpe.vocab_size} vocab size of length {len(bpe.vocab)}")
Nova = NovaLM(tokenizer= bpe, vocab_size= VOCAB_SIZE, embed_dim= EMBED_DIM, num_layers= NUM_LAYERS, num_heads= NUM_HEADS, max_seq_len= MAX_SEQ_LEN, rope_base= ROPE_BASE, dropout= DROPOUT).to(DEVICE)

if MODEL_SAVE_PATH.exists():
        print(f"Loading model from {MODEL_SAVE_PATH}")
        load_model(Nova, str(MODEL_SAVE_PATH))

print(f"{sum(p.numel() for p in Nova.parameters())} parameters in the model")
prompt = input("Enter a prompt: ")

messages = [{"role": "system", "content": "You are a knowledgeable assistant. Provide accurate, factual answers."},
            {"role": "user", "content": prompt}]

#output = Nova.chat(messages, TEMP, TOP_K, MAX_TOKENS,DEVICE)

output = Nova.generate(prompt, 0.01, TOP_K, MAX_TOKENS, DEVICE)
print(f"Nova: {output}")
