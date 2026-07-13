import torch
from torch import Tensor

class RoPE:
    def __init__(self, head_dim: int, max_seq_len: int, base: int = 10_000) -> None:
        assert head_dim % 2 == 0    # head_dim must be even to form rotation pairs
        
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        
        # θ_i = base ^(-2i/d) for i = 0, 1, ... d/2 - 1
        i = torch.arange(0, head_dim, 2).float()
        theta = base ** (-i / head_dim)
        
        pos = torch.arange(0, max_seq_len).float()
        angles = torch.outer(pos, theta)
        
        self.cos = torch.cos(angles)
        self.sin = torch.sin(angles)
        
    def apply_rotary(self, X: Tensor) -> Tensor:
        """
        Apply Rotary Position Embedding (RoPE) to the input tensor.

        RoPE rotates the input vectors in 2D subspaces based on their position.
        For each pair of dimensions (x1, x2), we apply:
            x1' = x1 * cos(θ) - x2 * sin(θ)
            x2' = x1 * sin(θ) + x2 * cos(θ)

        Args:
            X: Input tensor of shape (batch_size, T, n_heads, head_dim)

        Returns:
            Tensor with rotary embeddings applied, same shape as input
        """
        
        T = X.shape[1]
        cos = self.cos[:T].to(X.device).unsqueeze(0).unsqueeze(2)
        sin = self.sin[:T].to(X.device).unsqueeze(0).unsqueeze(2)
        
        x1, x2 = X.chunk(2, dim=-1)
        
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin - x2 * cos
        
        return torch.cat([x1_rot, x2_rot], dim=-1)