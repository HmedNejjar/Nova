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
        # Angles are built in float64 so pos * θ carries no fp32 rounding, then stored as fp32
        i = torch.arange(0, head_dim, 2, dtype=torch.float64)
        theta = base ** (-i / head_dim)
        
        pos = torch.arange(0, max_seq_len, dtype=torch.float64)
        angles = torch.outer(pos, theta).float()
        
        # Register the computed values as buffers
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        
    def forward(self, seq_len: int, position_ids: Tensor | None = None, offset: int = 0, dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
        """
        Look up cos/sin for the current positions, once per forward pass; every layer reuses them.

        Args:
            seq_len: Number of new tokens in the input
            position_ids: Optional (batch_size, seq_len) positions; overrides offset
            offset: Offset for the position indices, useful for caching in inference
            dtype: Dtype Q/K will have (bf16 under autocast), so the rotation never upcasts to fp32

        Returns:
            (cos, sin), each broadcastable to (batch_size, seq_len, n_heads, head_dim // 2)
        """
        if position_ids is None:
            cos = self.cos[offset: offset + seq_len].unsqueeze(0).unsqueeze(2)
            sin = self.sin[offset: offset + seq_len].unsqueeze(0).unsqueeze(2)
        else:
            position_ids = position_ids.to(self.cos.device)
            cos = self.cos[position_ids].unsqueeze(2)
            sin = self.sin[position_ids].unsqueeze(2)
        
        return cos.to(dtype), sin.to(dtype)

def apply_rotary(X: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """
    Apply Rotary Position Embedding (RoPE) to the input tensor.

    RoPE rotates the input vectors in 2D subspaces based on their position.
    For each pair of dimensions (x1, x2), we apply:
        x1' = x1 * cos(θ) - x2 * sin(θ)
        x2' = x1 * sin(θ) + x2 * cos(θ)

    Args:
        X: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
        cos, sin: From RoPE.forward(), already in X's dtype

    Returns:
        Tensor with rotary embeddings applied, same shape as input
    """
    x1, x2 = X.chunk(2, dim=-1)
    
    x1_rot = x1 * cos - x2 * sin
    x2_rot = x1 * sin + x2 * cos
    
    return torch.cat([x1_rot, x2_rot], dim=-1).type_as(X)