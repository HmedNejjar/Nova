import sys
from pathlib import Path
ROOT = Path(__file__).parent
sys.path.insert(1, str(ROOT))

import yaml
import torch
from GPT.Nova import NovaLM
from Preprocess.tokenizer import BPE

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

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

bpe = BPE(vocab_size= VOCAB_SIZE, savepath= SAVEPATH)
Nova = NovaLM(tokenizer= bpe, vocab_size= VOCAB_SIZE, embed_dim= EMBED_DIM, num_layers= NUM_LAYERS, num_heads= NUM_HEADS, max_seq_len= MAX_SEQ_LEN, rope_base= ROPE_BASE, dropout= DROPOUT).to(DEVICE)

if MODEL_SAVE_PATH.exists():
        print(f"Loading model from {MODEL_SAVE_PATH}")
        Nova.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))

prompt = input("Enter a prompt: ")

output = Nova.chat(prompt, TEMP, TOP_K, 60, DEVICE)
print(output)
