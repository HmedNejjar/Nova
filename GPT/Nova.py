import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
from torch import nn, Tensor
from GPT.decoder import Decoder
from Preprocess.tokenizer import BPE

class NovaLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, rope_base: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # integer token IDs -> dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = Decoder(embed_dim, num_layers, num_heads, max_seq_len, rope_base, dropout)
        
        # pre-final-projection norm
        self.final_norm = nn.LayerNorm(embed_dim)
        # hidden vectors -> vocab-sized logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias = False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, X: Tensor, cache_list: list[dict] | None) -> tuple[Tensor, list[dict]]:
        # Embedding the tokens into vectors
        X = self.token_embedding(X)
        # Pass it through the decoder
        X, new_cache_list = self.decoder(X, cache_list)
        # Normalization before logits computation
        X = self.final_norm(X)
        # Compute logits
        logits = self.lm_head(X)
        
        return (logits, new_cache_list)
    
    @torch.no_grad()
    def generate(self, tokenizer: BPE, prompt: str, max_new_tokens: int, device: str) -> str:
        """
        Generate text given a prompt using greedy decoding.
        Args:
            tokenizer: BPE tokenizer instance
            prompt: Input string prompt
            max_new_tokens: Maximum number of new tokens to generate
        Returns:
            Generated text as a string
        """
        self.eval()
        
        # Encode the prompt
        input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
        
        cache_list = None
        generated_ids = input_ids
        EOS_ID = tokenizer.vocab["<eos>"]
        
        # First pass — getting full cache list and first logits
        logits, cache_list = self.forward(generated_ids, cache_list)
        
        # Generation loop
        for _ in range(max_new_tokens):
            if generated_ids.shape[1] >= self.max_seq_len:
                break
            
            # Get the next token
            next_token_logits = logits[:, -1, :]
            next_token = next_token_logits.argmax(dim= -1, keepdim= True)
            
            generated_ids = torch.cat([generated_ids, next_token], dim= 1)
            
            # If EOS token is generated, stop generation
            if next_token.item() == EOS_ID:
                break
            
            # Forward pass to get logits for the next token
            logits, cache_list = self.forward(next_token, cache_list)
        
        # Decode the generated token IDs back to text
        generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist())
        return generated_text