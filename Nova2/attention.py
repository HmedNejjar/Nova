import torch
from torch import nn, Tensor
from .rope import RotaryEmbedding

class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 max_seq_len: int, rope_base: float, dropout: float, bias: bool = False) -> None:
        super().__init__()
        if embed_dim != num_heads * head_dim:
            raise ValueError("embed_dim must equal num_heads * head_dim")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_repeat = num_heads // num_kv_heads
        self.dropout = dropout
        self.q_proj = nn.Linear(embed_dim, num_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, num_kv_heads * head_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.rope = RotaryEmbedding(head_dim, max_seq_len, rope_base)

    def _repeat_kv(self, x: Tensor) -> Tensor:
        return x if self.kv_repeat == 1 else x.repeat_interleave(self.kv_repeat, dim=1)

    def forward(self, x: Tensor, past_key_value=None, use_cache: bool = False):
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_kv_heads, self.head_dim).transpose(1, 2)
        past_len = 0 if past_key_value is None else past_key_value[0].shape[-2]
        q, k = self.rope(q, past_len), self.rope(k, past_len)
        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=-2)
            v = torch.cat((past_key_value[1], v), dim=-2)
        present = (k, v) if use_cache else None
        k, v = self._repeat_kv(k), self._repeat_kv(v)
        # SDPA provides an efficient CUDA attention kernel on supported PyTorch/A100 builds.
        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0,
            is_causal=(past_key_value is None)
        )
        y = y.transpose(1, 2).contiguous().view(b, t, self.num_heads * self.head_dim)
        return self.out_proj(y), present
