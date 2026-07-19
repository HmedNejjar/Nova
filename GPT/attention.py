import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
import torch.nn as nn
from torch import Tensor

from Preprocess.pos_embed import RoPE

class BatchedMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, max_seq_len: int, rope_base: int) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Computation of Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj =  nn.Linear(embed_dim, embed_dim)
        
        # Initialize RoPE instance
        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=max_seq_len, base= rope_base)
        
        #Computation of attention output
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, X: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        """
        Args:
            X: Input tensor of shape (batch_size, seq_len T, embed_dim d)
            cache: KV cache stored in a dict if available

        Returns:
            Tensor of shape (batch_size, seq_len T, embed_dim d) after applying multi-head attention
            New cache containing all values of K and V
        """
        B, T, d = X.shape
        
        # Compute Q, K, V
        Q, K, V = self.q_proj(X), self.k_proj(X), self.v_proj(X)
        
        # Reshape Q, K an V for RoPE and attention
        # Transpose to (B, n_heads, T, head_dim) so matmul batches over (B, n_heads)
        Q = Q.view(B, T, self.num_heads, self.head_dim)
        K = K.view(B, T, self.num_heads, self.head_dim)
        V = V.view(B, T, self.num_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        offset = cache["K"].shape[2] if cache is not None else 0
        
        Q = self.rope.apply_rotary(Q, offset= offset)
        K = self.rope.apply_rotary(K, offset= offset)
        
        # Transpose to (B, n_heads, T, head_dim) so matmul batches over (B, n_heads)
        Q = Q.transpose(1, 2)  # (B, n_heads, T, head_dim)
        K = K.transpose(1, 2)  # (B, n_heads, T, head_dim)
        V = V.transpose(1, 2)  # (B, n_heads, T, head_dim)
        
        # Concat the cache for inference
        if cache is not None:
            K = torch.cat([cache["K"], K], dim= 2) # Concat on T dimension
            V = torch.cat([cache["V"], V], dim= 2) # Concat on T dimension
            
        new_cache = {'K': K,
                     'V': V}
        
        # Compute score
        scores: Tensor = (Q @ K.transpose(-2, -1)) / self.head_dim ** 0.5
        
        # Set a general causal mask using Q_len/K_len instead of classic (T, T)
        Q_len = Q.shape[2]
        K_len = K.shape[2]
        cache_offset = K_len - Q_len
        
        row = torch.arange(Q_len, device=X.device).unsqueeze(1)
        col = torch.arange(K_len, device=X.device).unsqueeze(0)
        
        causal_mask = (col > row + cache_offset).unsqueeze(0).unsqueeze(0)
        
        # Apply causal mask on scores
        scores = scores.masked_fill(causal_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Compute attention output
        attn_out = attn_weights @ V  # (B, n_heads, T, head_dim)
        
        # Transpose back to (B, T, n_heads, head_dim) and reshape to (B, T, d)
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(B, T, d)
        
        # Final linear projection
        return (self.out_proj(attn_out), new_cache)
        
        
        
        
        