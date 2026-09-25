from dataclasses import dataclass
import torch
from torch import nn, Tensor
from .attention import GroupedQueryAttention
from .normalization import RMSNorm

@dataclass
class Nova2Config:
    vocab_size: int = 150_000
    embed_dim: int = 2048
    num_layers: int = 28
    num_heads: int = 16
    hidden_dim: int = 5504
    num_kv_heads: int = 8
    head_dim: int = 128
    max_seq_len: int = 32768
    weight_tying: bool = True
    bias: bool = False
    dropout: float = 0.1
    eps: float = 1e-6
    rope_base: float = 100_000.0

class SwiGLU(nn.Module):
    def __init__(self, config: Nova2Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.embed_dim, config.hidden_dim, bias=config.bias)
        self.up_proj = nn.Linear(config.embed_dim, config.hidden_dim, bias=config.bias)
        self.down_proj = nn.Linear(config.hidden_dim, config.embed_dim, bias=config.bias)
    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

class Nova2Block(nn.Module):
    def __init__(self, config: Nova2Config) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.embed_dim, config.eps)
        self.ffn_norm = RMSNorm(config.embed_dim, config.eps)
        self.attn = GroupedQueryAttention(config.embed_dim, config.num_heads, config.num_kv_heads,
                                          config.head_dim, config.max_seq_len, config.rope_base,
                                          config.dropout, config.bias)
        self.ffn = SwiGLU(config)
        self.dropout = nn.Dropout(config.dropout)
    def forward(self, x: Tensor, past_key_value=None, use_cache: bool = False):
        y, present = self.attn(self.attn_norm(x), past_key_value, use_cache)
        x = x + self.dropout(y)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, present

class Nova2ForCausalLM(nn.Module):
    def __init__(self, config: Nova2Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.embed_dim)
        self.layers = nn.ModuleList(Nova2Block(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.embed_dim, config.eps)
        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        if config.weight_tying:
            self.lm_head.weight = self.token_embedding.weight
        self.apply(self._init_weights)
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, input_ids: Tensor, labels: Tensor | None = None, past_key_values=None, use_cache: bool = False):
        if input_ids.ndim != 2: raise ValueError("input_ids must have shape [batch, seq_len]")
        x = self.token_embedding(input_ids)
        presents = []
        for i, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[i]
            x, present = layer(x, past, use_cache)
            if use_cache: presents.append(present)
        logits = self.lm_head(self.final_norm(x))
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100)
        return {"loss": loss, "logits": logits, "past_key_values": presents if use_cache else None}
    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int, temperature: float = 0.7,
                 top_k: int = 6, repetition_penalty: float = 1.1, eos_token_id: int | None = None) -> Tensor:
        self.eval(); out = input_ids
        result = self(out, use_cache=True); logits, cache = result["logits"], result["past_key_values"]
        for _ in range(max_new_tokens):
            next_logits = logits[:, -1, :].clone()
            if repetition_penalty != 1.0:
                ids = torch.unique(out, dim=-1)
                vals = next_logits.gather(1, ids)
                vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
                next_logits.scatter_(1, ids, vals)
            if temperature <= 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                next_logits.div_(temperature)
                if top_k > 0:
                    values, indices = torch.topk(next_logits, min(top_k, next_logits.size(-1)), dim=-1)
                    filtered = torch.full_like(next_logits, float("-inf"))
                    filtered.scatter_(1, indices, values); next_logits = filtered
                next_token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            out = torch.cat((out, next_token), dim=1)
            if eos_token_id is not None and torch.all(next_token == eos_token_id): break
            result = self(next_token, past_key_values=cache, use_cache=True)
            logits, cache = result["logits"], result["past_key_values"]
        return out
    def num_parameters(self, trainable_only: bool = True) -> int:
        ps = (p for p in self.parameters() if p.requires_grad) if trainable_only else self.parameters()
        return sum(p.numel() for p in ps)
