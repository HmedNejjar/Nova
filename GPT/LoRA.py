import torch
import torch.nn as nn
from torch import Tensor

class LoRALinear(nn.Module):
    def __init__(self, base_layer: nn.Linear, d_in: int, d_out: int, rank: int, alpha: int, dropout: float, bias: bool = True) -> None:
        super().__init__()
        
        assert (rank, alpha) > (0, 0)
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Base linear layer
        self.base_linear = base_layer
        
        # LoRA layers for matrices A and B
        self.lora_A = nn.Linear(d_in, rank, bias=False)
        self.lora_B = nn.Linear(rank, d_out, bias=False)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # LoRA initialization
        nn.init.kaiming_uniform_(self.lora_A.weight, a= 5**0.5)
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze original weights
        for param in self.base_linear.parameters():
            param.requires_grad = False
            
    def forward(self, X: Tensor) -> Tensor:
        # Original linear transformation
        original_output = self.base_linear(X)
        
        # LoRA transformation
        lora_output = self.scaling * self.lora_B(self.dropout(self.lora_A(X)))  # Formula is: α/r (B@A)        
        # Combine outputs
        return original_output + lora_output