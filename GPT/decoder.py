import torch
import torch.nn as nn
from torch import Tensor
from GPT.attention import BatchedMultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, rope_base: int) -> None:
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        hidden_dim = int(2 * (4 * embed_dim) / 3)
        
        # Linear functions for FFN computation
        self.linear1 = nn.Linear(embed_dim, hidden_dim) # Gate projection
        self.linear2 = nn.Linear(hidden_dim, embed_dim) # project back to embed_dim
        self.linear3 = nn.Linear(embed_dim, hidden_dim) # Value projection
        
        # LayerNorms for pre-attention and pre-FFN
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Batched MHA instance
        self.MultiHeadAttention = BatchedMultiHeadAttention(embed_dim= embed_dim, num_heads= num_heads, max_seq_len= max_seq_len, rope_base= rope_base)
        
    def forward(self, X: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        # 1. Normalize X
        X_norm = self.norm1(X)
        
        # 2. Apply Multi-Head Attention
        attn_out, new_cache = self.MultiHeadAttention(X_norm, cache)
        
        # 3. Add residual connection
        X = X + attn_out
        
        # 4. Normalize again
        X_norm = self.norm2(X)
        
        # 5. Apply FFN
        ffn_out = self.FFN_SwiGLU(X_norm)
        
        # 6. Add residual connection
        X = X + ffn_out
        
        return (X, new_cache)
    
    def FFN_SwiGLU(self, x: Tensor) -> Tensor:
        """
        Feed ForwarD Network with SwiGLU activation function.

        Args:
            x: Input tensor of shape (batch_size, seq_len T, embed_dim d)
        Returns:
            Tensor of shape (batch_size, seq_len T, embed_dim d) after applying SwiGLU
        """
        gate = self.linear1(x)
        swish = gate * torch.sigmoid(gate)
        value = self.linear3(x)
        
        swiglu = swish * value
        
        return self.linear2(swiglu)

class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, rope_base: int) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList(DecoderBlock(embed_dim= embed_dim, num_heads= num_heads, max_seq_len= max_seq_len, rope_base= rope_base)
                                    for _ in range(num_layers))
        
    def forward(self, X: Tensor, cache_list: list[dict] | None) -> tuple[Tensor, list[dict]]:
        new_cache_list = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = cache_list[i] if cache_list is not None else None
            
            X, new_cache = block(X, layer_cache)
            new_cache_list.append(new_cache)
                
        return (X, new_cache_list)
    
    
        
        