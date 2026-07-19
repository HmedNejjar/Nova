import torch
from torch import nn, Tensor
from decoder import Decoder

class NovaLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, rope_base: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        
        # integer token IDs -> dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = Decoder(embed_dim, num_layers, num_heads, max_seq_len, rope_base)
        
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