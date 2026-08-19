import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
from torch import nn, Tensor
from GPT.decoder import Decoder
from Preprocess.tokenizer import BPE

class NovaLM(nn.Module):
    def __init__(self, tokenizer: BPE, vocab_size: int, embed_dim: int, num_layers: int, num_heads: int, max_seq_len: int, rope_base: int, dropout: float) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        # integer token IDs -> dense vectors
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder = Decoder(embed_dim, num_layers, num_heads, max_seq_len, rope_base, dropout)
        
        # pre-final-projection norm
        self.final_norm = nn.LayerNorm(embed_dim)
        # hidden vectors -> vocab-sized logits
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias = False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight
        
    def forward(self, X: Tensor, cache_list: list[dict] | None) -> tuple[Tensor, list[dict]]:
        # Embedding the tokens into vectors
        X = self.token_embedding(X)
        # Pass it through the decoder
        X, new_cache_list = self.decoder(X, cache_list)
        # Normalization before logits computation
        X = self.final_norm(X)
        # Compute logits
        logits = self.lm_head(X)
        
        return (logits, new_cache_list)
    
    @torch.no_grad()
    def generate(self, prompt: str, temperature: float, top_k: int, max_new_tokens: int, device: str) -> str:
        """
        Generate text given a prompt using temperature-scaled sampling and optional top-k filtering.
        Args:
            prompt: Input string prompt.
            temperature: Temperature for scaling logits before sampling.
            top_k: Number of highest-probability tokens to keep before sampling.
            max_new_tokens: Maximum number of new tokens to generate.
            device: Device to run inference on (e.g. 'cpu' or 'cuda').
        Returns:
            Generated text as a string.
        """
        self.eval()
        
        # Encode the prompt
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(device)
        
        cache_list = None
        generated_ids = input_ids
        EOS_ID = self.tokenizer.vocab["<eos>"]
        
        # First pass — getting full cache list and first logits
        logits, cache_list = self.forward(generated_ids, cache_list)
        
        # Generation loop
        for _ in range(max_new_tokens):
            if generated_ids.shape[1] >= self.max_seq_len:
                break
            
            # Get the next token scaled by temperature
            next_token_logits = logits[:, -1, :] / temperature
            
            if top_k > 0:
                # Keep only the top_k highest logits and set all others to -inf.
                # This implements top-k sampling by restricting the candidate tokens.
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim= -1)
                filtered_logits = torch.full_like(next_token_logits, float('-inf'))
                filtered_logits.scatter_(1, top_k_indices, top_k_values)
                
                # Use the filtered logits for the next sampling step.
                next_token_logits = filtered_logits
            
            # Convert logits to probabilities and sample one next token.
            probs = torch.softmax(next_token_logits, dim= -1)
            next_token = torch.multinomial(probs, num_samples= 1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim= 1)
            
            # If EOS token is generated, stop generation
            if next_token.item() == EOS_ID:
                break
            
            # Forward pass to get logits for the next token
            logits, cache_list = self.forward(next_token, cache_list)
        
        # Decode the generated token IDs back to text
        generated_text = self.tokenizer.decode(generated_ids.squeeze(0).tolist())
        return generated_text
    
    @torch.no_grad()
    def chat(self, messages: list[dict], temperature: float, top_k: int, max_new_tokens: int, device: str) -> str:
        ROLE_TAGS = {"user": "<|user|>", "assistant": "<|assistant|>", "system": "<|system|>"}
        max_prompt_len = max(self.max_seq_len - max_new_tokens, 1)
    
        # Keep system instructions separate so history can be trimmed first.
        system_turns, convo_turns = [], []
        for m in messages:
            role = m.get("role", "user")
            tag = ROLE_TAGS.get(role, "<|user|>")
            encoded = self.tokenizer.encode(f"{tag} {m.get('content', '')} ")
            (system_turns if role == "system" else convo_turns).append(encoded)
    
        assistant_lead_in = self.tokenizer.encode("<|assistant|> ")
        bos_ids = self.tokenizer.encode("<bos> ")
    
        # Reserve space for BOS, system turns, and the assistant marker.
        budget = max_prompt_len - len(bos_ids) - sum(len(t) for t in system_turns) - len(assistant_lead_in)
        while convo_turns and sum(len(t) for t in convo_turns) > budget:
            convo_turns.pop(0)
    
        prompt_ids = list(bos_ids)
        for t in system_turns + convo_turns:
            prompt_ids.extend(t)
        prompt_ids.extend(assistant_lead_in)
    
        # This is a final guard for oversized system prompts; keep the newest suffix.
        if len(prompt_ids) > max_prompt_len:
            prompt_ids = prompt_ids[-max_prompt_len:]
    
        prompt = self.tokenizer.decode(prompt_ids)
        output = self.generate(prompt, temperature, top_k, max_new_tokens, device)
    
        # Return only the assistant's first response, stopping at any new turn marker.
        assistant_text = output.rsplit("<|assistant|>", 1)[1] if "<|assistant|>" in output else output
        assistant_text = assistant_text.replace("<bos>", "").replace("<eos>", "")
        for tag in ("<|user|>", "<|assistant|>", "<|system|>"):
            if tag in assistant_text:
                assistant_text = assistant_text.split(tag, 1)[0]
        return assistant_text.strip()