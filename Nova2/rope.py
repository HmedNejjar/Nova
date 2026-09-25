import torch
from torch import nn, Tensor

class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, base: float = 100_000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(positions, inv_freq)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, offset: int = 0) -> Tensor:
        seq_len = x.shape[-2]
        end = offset + seq_len
        if end > self.cos.shape[0]:
            raise ValueError(f"RoPE position {end} exceeds max_seq_len={self.cos.shape[0]}")
        cos = self.cos[offset:end].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        sin = self.sin[offset:end].to(device=x.device, dtype=x.dtype)[None, None, :, :]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1).flatten(-2)
