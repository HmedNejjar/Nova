import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import cast
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from GPT.attention import GroupedQueryAttention as GQA
from Preprocess.pos_embed import RoPE

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
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, num_kv_heads: int, hidden_dim: int, dropout: float = 0.1, eps: float = 1e-6, bias: bool = False) -> None:
        super().__init__()

        self.attn_norm = RMSNorm(d_model= embed_dim, eps= eps)
        self.ffn_norm = RMSNorm(d_model= embed_dim, eps= eps)
        self.gqa = GQA(embed_dim, num_heads, head_dim, num_kv_heads, bias)
        
        # SwiGLU Linear layers
        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias= bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias= bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias= bias)

        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, X: Tensor, rope: tuple[Tensor, Tensor], cache: dict | None = None, attn_mask: Tensor | None = None) -> tuple[Tensor, dict]:
        # 1. Pre-Attention normalization
        X_norm = self.attn_norm(X)

        # 2. Apply GQA
        attn_out, new_cache = self.gqa(X_norm, rope, cache, attn_mask)
        
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
    def __init__(self, embed_dim: int, num_layers: int, num_heads: int, head_dim: int, num_kv_heads: int, max_seq_len: int, hidden_dim: int, rope_base: int = 10_000, dropout: float = 0.1, eps: float = 1e-6, bias: bool = False, checkpoint_layers: int = 0) -> None:
        super().__init__()

        self.num_layers = num_layers
        # Activation checkpointing for the first `checkpoint_layers` blocks (0 = off, num_layers = all):
        # each checkpointed block saves its activations' memory but reruns its forward during backward
        self.checkpoint_layers = checkpoint_layers

        # One RoPE table shared by every layer
        self.rope = RoPE(head_dim=head_dim, max_seq_len=max_seq_len, base=rope_base)

        self.blocks = nn.ModuleList(DecoderBlock(embed_dim, num_heads, head_dim, num_kv_heads, hidden_dim, dropout, eps, bias)
                                    for _ in range(num_layers))

    def forward(self, X: Tensor, cache_list: list[dict] | None, position_ids: Tensor | None = None, attn_mask: Tensor | None = None) -> tuple[Tensor, list[dict]]:
        new_cache_list = []

        use_checkpoint = self.training and cache_list is None

        # cos/sin are looked up once here and shared by all layers, already in the
        # dtype Q/K will have under autocast so the rotation never upcasts to fp32
        offset = cache_list[0]["K"].shape[2] if cache_list is not None else 0
        dtype = torch.get_autocast_dtype(X.device.type) if torch.is_autocast_enabled(X.device.type) else X.dtype
        rope = self.rope.cos_sin(X.shape[1], position_ids, offset, dtype)

        for i, block in enumerate(self.blocks):
            layer_cache = cache_list[i] if cache_list is not None else None

            if use_checkpoint and i < self.checkpoint_layers:
                X, new_cache = cast(tuple[Tensor, dict], checkpoint(block, X, rope, layer_cache, attn_mask, use_reentrant=False),)
            else:
                X, new_cache = block(X, rope, layer_cache, attn_mask)
            new_cache_list.append(new_cache)
                
        return (X, new_cache_list)
    
    
        
        