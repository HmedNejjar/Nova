import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
from torch import nn, Tensor

from GPT.decoder import RMSNorm, Decoder
from Preprocess.Tokenizer.tokenizer import BPE

class NovaLM(nn.Module):
    def __init__(self, config: dict) -> None:
        """
        Args:
            tokenizer: An instance of the BPE tokenizer.
            config: A dictionary containing model and tokenizer hyperparameters:
                -vocab_size: int
                -embed_dim: int
                -num_layers: int
                -num_heads: int
                -head_dim: int
                -num_kv_heads: int
                -max_seq_len: int
                -hidden_dim: int
                -rope_base: int
                -dropout: float
                -eps: float
                -bias: bool
        """
        
        super().__init__()
        model_config = config["Model"]
        tokenizer_config = config["Tokenizer"]
        tokenizer_path = tokenizer_config["savepath"]
        
        self.vocab_size = tokenizer_config["vocab_size"]
        self.special_tokens = tokenizer_config["special_tokens"]
        
        self.embed_dim = model_config["embed_dim"]
        self.num_layers = model_config["num_layers"]
        self.num_heads = model_config["num_heads"]
        self.head_dim = model_config["head_dim"]
        self.num_kv_heads = model_config["num_kv_heads"]
        self.max_seq_len = model_config["max_seq_len"]
        self.hidden_dim = model_config["hidden_dim"]
        self.rope_base = int(model_config["rope_base"])
        self.dropout = float(model_config["dropout"])
        self.eps = float(model_config["eps"])
        self.bias = bool(model_config["bias"])


        # Configure model tokenizer
        self.tokenizer = BPE(self.vocab_size, tokenizer_path)
        
        # Token embedding layer
        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # Decoder blocks
        # Activation checkpointing is a training-memory knob, so it lives under Train
        checkpoint_layers = int(config.get("Train", {}).get("grad_checkpoint_layers", 0))
        self.decoder = Decoder(self.embed_dim, self.num_layers, self.num_heads, self.head_dim, self.num_kv_heads, self.max_seq_len, self.hidden_dim, self.rope_base, self.dropout, self.eps, self.bias, checkpoint_layers)
        
        # Final normalization layer
        self.final_norm = RMSNorm(d_model=self.embed_dim, eps=self.eps)
        
        # Language modeling head (embed_dim -> vocab_size)
        self.lm_head = nn.Linear(self.embed_dim, self.vocab_size, bias= self.bias)
        
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
        self.apply(self._init_weights)
        # scale down the projections feeding the residual stream
        for name, p in self.named_parameters():
            if name.endswith(("out_proj.weight", "down_proj.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.num_layers) ** 0.5)
                
    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        
    def forward(self, X: Tensor, cache_list: list[dict] | None = None, position_ids: Tensor | None = None, attn_mask: Tensor | None = None, return_hidden: bool = False) -> tuple[Tensor, list[dict]]:
        # Embedding the tokens into vectors
        X = self.token_embedding(X)
        # Pass it through the decoder
        X, new_cache_list = self.decoder(X, cache_list, position_ids, attn_mask)
        # Normalization before logits computation
        X = self.final_norm(X)
        # Training computes lm_head + loss in chunks (Training/eval.py::chunked_lm_loss)
        # so the full (batch_size, seq_len, vocab) logits never exist at once
        if return_hidden:
            return (X, new_cache_list)
        # Compute logits
        logits = self.lm_head(X)
        return (logits, new_cache_list)
    
    @property
    def parameter_summary(self) -> str:
        summary = "\n".join(f"Parameter {name}: {value}" for name, value in vars(self).items() if not name.startswith("_"))
        total_params = sum(parameter.numel() for parameter in self.parameters())
        return f"{summary}\nTotal parameters: {total_params}"
    
    @torch.no_grad()
    def generate(self, prompt: str | list[int], max_new_tokens: int = 128, temperature: float = 1.0, top_k: int = 0, top_p: float = 1.0, repetition_penalty: float = 1.0,
                 stop_tokens: tuple[str, ...] = ("<eos>",), device: str | torch.device | None = None, return_prompt: bool = True) -> str:
        """
        Autoregressive generation with a KV cache.

        Args:
            prompt: Text, or token ids (chat() passes ids so nothing is re-tokenized).
            max_new_tokens: Upper bound on generated tokens.
            temperature: 0 = greedy (argmax); otherwise logits are divided by it before sampling.
            top_k: Keep only the k most likely tokens (0 = off).
            top_p: Nucleus sampling, keep the smallest set with cumulative prob >= top_p (1.0 = off).
            repetition_penalty: >1 discourages tokens already in the prompt/output (1.0 = off).
            stop_tokens: Special tokens that end generation; they are not included in the output.
            device: Defaults to the model's device.
            return_prompt: Include the prompt in the returned text (False = only the continuation).
        Returns:
            Decoded text.
        """
        assert max_new_tokens <= self.max_seq_len, "max_new_tokens must be <= max_seq_len"
        was_training = self.training
        self.eval()
        
        device = torch.device(device) if device is not None else next(self.parameters()).device
        
        # Encode the prompt
        input_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else [int(token) for token in prompt]
        
        if not input_ids:
            raise ValueError("Cannot generate from empty prompt")
        
        # Positions past max_seq_len have no RoPE entry: keep the newest prompt tokens
        input_ids = input_ids[-max(self.max_seq_len - max_new_tokens, 1):]
        
        vocab = self.tokenizer.vocab
        stop_token_ids = [vocab[token] for token in stop_tokens]
        generated_ids = []
        
        try:
            # Prefill: one pass over the prompt builds the cache; only the last position needs logits
            hidden, cache = self.forward(torch.tensor(input_ids).unsqueeze(0).to(device), None, None, None, return_hidden=True)
            
            for _ in range(max_new_tokens):
                # Compute logits for the last position and sample
                logits = self.lm_head(hidden[:, -1]).float()[0]
                
                # Sample the next token from those logits
                next_id = self._sample(logits, input_ids + generated_ids, temperature, top_k, top_p, repetition_penalty)
                
                # Check stop condition
                if next_id in stop_token_ids:
                    break
                # Record the token we just generated
                generated_ids.append(next_id)
                
                if len(input_ids) + len(generated_ids) >= max_new_tokens:
                    break
                
                # Feed that token back through the model to get the next logits
                hidden, cache = self.forward(torch.tensor([[next_id]], device=device), cache, return_hidden=True)
                
        finally:
            # Callers (e.g. eval during training) get the model back in the mode they left it
            self.train(was_training)

        return self.tokenizer.decode((input_ids + generated_ids) if return_prompt else generated_ids)

                
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 512, temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.1, 
             system_prompt: str | None = None, device: str | torch.device | None = None, return_thinking: bool = False) -> str | tuple[str, str]:
        """
        Reply to a conversation using exactly the Phase 3 training format:
            <bos><system>S</system><user>U</user><assistant>A</assistant> ... <user>U</user><assistant>
        Generation stops at </assistant> or <eos>.

        Args:
            messages: [{"role": "system" | "user" | "assistant", "content": str}, ...], ending with a user turn.
            system_prompt: Overrides any system message. Use the same prompt Phase 3 was trained with.
            return_thinking: Return (thinking, answer) instead of only the answer.
        
        Returns:
            Decoded text.
        """
        SPECIAL_TOKENS = self.special_tokens
        
        # Get the system prompt
        sys_prompt = system_prompt or next((msg["content"] for msg in messages if msg.get("role") == "system"), None)
        
        turns = [msg for msg in messages if msg.get("role") in ("user", "assistant")]
        
        # Check the last turn is a user turn
        if not turns or turns[-1]["role"] != "user":
            raise ValueError("Chat messages must end with a user turn")
        
        # Reference the encoder
        encode = self.tokenizer.encode
        
        head = encode(SPECIAL_TOKENS["bos"]) + (encode(f"{SPECIAL_TOKENS['system']}{sys_prompt}{SPECIAL_TOKENS['end_system']}") if sys_prompt else [])
        
        # The reply is generated right after the assistant opening tag
        lead_in = encode(SPECIAL_TOKENS["assistant"])
        
        # Opening / closing tags per role
        role_tags = {
            "user": (SPECIAL_TOKENS["user"], SPECIAL_TOKENS["end_user"]),
            "assistant": (SPECIAL_TOKENS["assistant"], SPECIAL_TOKENS["end_assistant"]),
        }
        
        # Encode every turn once, keeping its role for trimming
        encoded_turns = []
        for msg in turns:
            open_tag, close_tag = role_tags[msg["role"]]
            encoded_turns.append((msg["role"], encode(f"{open_tag}{msg['content']}{close_tag}")))
        
        # Drop the oldest turns until the prompt leaves room for the reply.
        # The history must start on a user turn, and the last user message is always kept.
        budget = self.max_seq_len - max_new_tokens - len(head) - len(lead_in)
        while len(encoded_turns) > 1 and (sum(len(ids) for _, ids in encoded_turns) > budget or encoded_turns[0][0] != "user"):
            encoded_turns.pop(0)
        
        # Build the prompt ids directly, so nothing is decoded and re-tokenized
        prompt_ids = head + [tok for _, ids in encoded_turns for tok in ids] + lead_in
        
        # Generate the reply
        reply = self.generate(prompt_ids, max_new_tokens= max_new_tokens, temperature= temperature, top_k= top_k, top_p= top_p, repetition_penalty= repetition_penalty,
                              stop_tokens= (SPECIAL_TOKENS["end_assistant"], SPECIAL_TOKENS["eos"]), device= device, return_prompt= False)
        
        # A thinking reply looks like "answer"
        thinking, answer = "", reply
        if SPECIAL_TOKENS["end_thinking"] in reply:
            thinking, answer = reply.split(SPECIAL_TOKENS["end_thinking"], 1)
            thinking = thinking.replace(SPECIAL_TOKENS["thinking"], "")
        
        return (thinking.strip(), answer.strip()) if return_thinking else answer.strip()

    @staticmethod
    def _sample(logits: Tensor, seen: list[int], temperature: float, top_k: int, top_p: float,
                repetition_penalty: float) -> int:
        """Pick the next token id from one position's logits (vocab,)."""
        if repetition_penalty != 1.0 and seen:
            idx = torch.tensor(sorted(set(seen)), device=logits.device)
            vals = logits[idx]
            # Divide positive logits, multiply negative ones: both make the token less likely
            logits[idx] = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)

        if temperature <= 0.0:
            return int(logits.argmax())
        logits = logits / temperature

        if 0 < top_k < logits.numel():
            kth = torch.topk(logits, top_k).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            # Drop tokens once the mass BEFORE them already reaches top_p (always keeps the top token)
            sorted_logits[(probs.cumsum(-1) - probs) >= top_p] = float("-inf")
            logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)

        return int(torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1))