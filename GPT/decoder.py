import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import BlockMask
from torch import Tensor
from GPT.attention import GroupedQueryAttention as GQA

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        self.scale = nn.Parameter(torch.ones(d_model))
        
    def forward(self, X: Tensor) -> Tensor:
        # Formula for RMSNorm: x / sqrt(mean(x²) + eps)
        norm = torch.rsqrt(X.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.scale * X * norm

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, num_kv_heads: int, max_seq_len: int, hidden_dim: int, rope_base: int = 10_000, dropout: float = 0.1, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        
        self.attn_norm = RMSNorm(d_model= embed_dim, eps= eps)
        self.ffn_norm = RMSNorm(d_model= embed_dim, eps= eps)
        self.gqa = GQA(embed_dim, num_heads, head_dim, num_kv_heads, max_seq_len, rope_base, bias)
        
        # SwiGLU Linear layers
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias= bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias= bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias= bias)

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, X: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        # 1. Pre-Attention normalization
        X_norm = self.attn_norm(X)
        
        # 2. Apply GQA
        attn_out, new_cache = self.gqa(X_norm, cache)
        
        # 3. Add residual connection with dropout
        X = X + self.dropout(attn_out)
        
        # 4. Pre-FFN normalization
        X_norm = self.ffn_norm(X)
        
        # 5. Apply FFN
        ffn_out = self.FFN_SwiGLU(X_norm)
        
        # 6. Add residual connection with dropout
        X = X + self.dropout(ffn_out)
        
        return (X, new_cache)
    
    def FFN_SwiGLU(self, x: Tensor) -> Tensor:
        """
        Feed Forward Network with SwiGLU activation function.

        Args:
            x: Input tensor of shape (batch_size, seq_len, embed_dim )
        Returns:
            Tensor of shape (batch_size, seq_len , embed_dim ) after applying SwiGLU
        """
        gated = F.silu(self.gate_proj(x)) * self.up_proj(x)
        return self.down_proj(gated)

class Decoder(nn.Module):
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, head_dim: int, num_kv_heads: int, max_seq_len: int, hidden_dim: int, rope_base: int = 10_000, dropout: float = 0.1, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()
        
        self.num_layers = num_layers
        
        self.blocks = nn.ModuleList(DecoderBlock(embed_dim, num_heads, head_dim, num_kv_heads, max_seq_len, hidden_dim, rope_base, dropout, eps, bias)
                                    for _ in range(num_layers))
        
    def forward(self, X: Tensor, cache_list: list[dict] | None) -> tuple[Tensor, list[dict]]:
        new_cache_list = []
        
        for i, block in enumerate(self.blocks):
            layer_cache = cache_list[i] if cache_list is not None else None
            
            X, new_cache = block(X, layer_cache)
            new_cache_list.append(new_cache)
                
        return (X, new_cache_list)
    
    
        
        