import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
import torch.nn as nn
from torch.nn.functional import scaled_dot_product_attention as Flash_Attention
from torch import Tensor

from Preprocess.pos_embed import RoPE

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, head_dim: int, num_kv_heads: int, max_seq_len: int, rope_base: int, bias: bool) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.embed_dim = embed_dim
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_group = num_heads // num_kv_heads
        
        # Computation of Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias= bias)
        self.v_proj =  nn.Linear(embed_dim, num_kv_heads * self.head_dim, bias= bias)
        
        # Initialize RoPE instance
        self.rope = RoPE(head_dim=self.head_dim, max_seq_len=max_seq_len, base= rope_base)
        
        #Computation of attention output
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias= bias)
        
    def forward(self, X: Tensor, cache: dict | None = None) -> tuple[Tensor, dict]:
        """
        Args:
            X: Input tensor of shape (batch_size, seq_len T, embed_dim d)
            cache: KV cache stored in a dict if available

        Returns:
            Tensor of shape (batch_size, seq_len T, embed_dim d) after applying multi-head attention
            New cache containing all values of K and V
        """
        batch_size, seq_len, _ = X.shape
        
        # Compute Q, K, V
        Q = self.q_proj(X).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_proj(X).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        V = self.v_proj(X).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        
        # Apply RoPE to Q and K
        offset = cache["K"].shape[2] if cache is not None else 0
        
        Q = self.rope.apply_rotary(Q, offset= offset)
        K = self.rope.apply_rotary(K, offset= offset)
        
        # Transpose Q, K, V
        Q = Q.transpose(1,2) # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1,2) # (batch_size, num_kv_heads, seq_len, head_dim)
        V = V.transpose(1,2) # (batch_size, num_kv_heads, seq_len, head_dim)
        
        # Concat the cache for inference
        if cache is not None:
            K = torch.cat([cache["K"], K], dim= 2) # Concat on seq_len dimension
            V = torch.cat([cache["V"], V], dim= 2) # Concat on seq_len dimension
            
        new_cache = {'K': K,
                     'V': V}
        
        # Compute score
        scores: Tensor = (Q @ K.transpose(-2, -1)) / self.head_dim ** 0.5
        
        # Apply a causal mask that accounts for cached keys.
        Q_len = Q.size(2)
        K_len = K.size(2)
        past_len = K_len - Q_len
        if past_len < 0:
            raise ValueError("kv_cache cannot contain fewer tokens than the current input")
        
        if past_len == 0:
            attn_out = Flash_Attention(Q, K, V, is_causal= True, enable_gqa= True)
        else:
            # With cached keys, queries start after the cached prefix, so the
            # built-in causal mask cannot represent their absolute positions.
            query_positions = torch.arange(Q_len, device=X.device).unsqueeze(1)
            # Each query may attend to keys up to its position in the full sequence.
            key_positions = torch.arange(K_len, device=X.device).unsqueeze(0)
            causal_mask = key_positions <= past_len + query_positions
            
            # Use the explicit mask so cached tokens remain visible while future
            # tokens are still hidden.
            attn_out = Flash_Attention(Q, K, V, attn_mask= causal_mask, enable_gqa= True)
        
        # Transpose back and reshape
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final linear projection
        attn_out = self.out_proj(attn_out)
        
        return (attn_out, new_cache)
        
        
        
        
        