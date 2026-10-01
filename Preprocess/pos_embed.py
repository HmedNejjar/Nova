import torch
import torch.nn as nn
from torch import Tensor

class RoPE(nn.Module):
    cos: Tensor
    sin: Tensor

    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000) -> None:
        super().__init__()
        assert head_dim % 2 == 0    # head_dim must be even to form rotation pairs
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # θ_i = base ^(-2i/d) for i = 0, 1, ... d/2 - 1
        i = torch.arange(0, head_dim, 2).float()
        theta = base ** (-i / head_dim)
        
        pos = torch.arange(0, max_seq_len).float()
        angles = torch.outer(pos, theta)
        
        # Register the computed values as buffers
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        
    def cos_sin(self, seq_len: int, position_ids: Tensor | None = None, offset: int = 0, dtype: torch.dtype | None = None) -> tuple[Tensor, Tensor]:
        """
        Look up the rotation angles once per forward pass; every layer reuses them.

        Args:
            seq_len: Number of positions in the current input
            position_ids: (batch_size, seq_len) per-token positions, or None for offset..offset+seq_len
            offset: Offset for the position indices, useful for caching in inference
            dtype: Dtype Q/K will have (the autocast dtype in mixed precision), so rotating never upcasts

        Returns:
            (cos, sin), each broadcastable to (batch_size, seq_len, n_heads, head_dim // 2)
        """
        if position_ids is None:
            cos = self.cos[offset: offset + seq_len].unsqueeze(0)
            sin = self.sin[offset: offset + seq_len].unsqueeze(0)
        else:
            cos = self.cos[position_ids]
            sin = self.sin[position_ids]

        dtype = dtype or cos.dtype
        return cos.unsqueeze(2).to(dtype), sin.unsqueeze(2).to(dtype)

    @staticmethod
    def apply_rotary(X: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """
        Apply Rotary Position Embedding (RoPE) to the input tensor.

        RoPE rotates the input vectors in 2D subspaces based on their position.
        For each pair of dimensions (x1, x2), we apply:
            x1' = x1 * cos(θ) - x2 * sin(θ)
            x2' = x1 * sin(θ) + x2 * cos(θ)

        Args:
            X: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
            cos, sin: From cos_sin()

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        x1, x2 = X.chunk(2, dim=-1)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        
        return torch.cat([x1_rot, x2_rot], dim=-1).type_as(X)