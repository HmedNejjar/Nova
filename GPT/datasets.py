import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import torch
from torch.utils.data import Dataset
from Preprocess.tokenizer import BPE

class VocabDataset(Dataset):
    def __init__(self, tokenized_ds: list[int], seq_len: int, stride_coeff: int = 2) -> None:
        super().__init__()
        
        self.tokenized_data = torch.tensor(tokenized_ds)
        self.seq_len = seq_len
        self.stride = seq_len // stride_coeff
        
        # Create list of starting indices for each window
        self.start_indices = list(range(0, len(self.tokenized_data) - seq_len, self.stride))
        assert len(self.start_indices) > 0, f"seq_len={seq_len} too large for corpus of length {len(self.tokenized_data)}"
        
    def __len__(self) -> int:
        return len(self.start_indices)
    
    def __getitem__(self, idx: int) -> tuple:
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.seq_len
        
        input_seq = self.tokenized_data[start_idx:end_idx]
        target_seq = self.tokenized_data[start_idx + 1:end_idx + 1]
        
        return (input_seq, target_seq)
    
class ChatBotDataset(Dataset):
    def __init__(self, conversations: list[str], tokenizer: BPE, seq_len: int) -> None:
        super().__init__()
        
        self.seq_len = seq_len
        self.examples = []
        
        USER_ID, ASSISTANT_ID = tokenizer.vocab["<|user|>"], tokenizer.vocab["<|assistant|>"]
        PAD_ID = tokenizer.vocab["<pad>"]
        
        for convo in conversations:
            # Tokenize the conversation and add special tokens
            tokenized_convo = tokenizer.encode(convo)
            
            # Truncate or pad to fit seq_len (+1 for shift (target))
            tokenized_convo = self._truncate_or_pad(tokenized_convo, seq_len + 1, PAD_ID)
            
            # Create input and target sequences
            input_seq = tokenized_convo[:-1]  # All but the last token
            target_seq = tokenized_convo[1:]  # All but the first token
            
            # Apply assistant-only masking
            masked_target = target_seq.copy()
            masking = True # always mask by default
            for i in range(len(target_seq)):
                token = input_seq[i]
                
                if token == USER_ID:
                    masking = True 
                elif token == ASSISTANT_ID:
                    masking = False  
                
                if masking or token == PAD_ID:
                    masked_target[i] = -100  # Mask user tokens and padding tokens
                    
            self.examples.append((torch.tensor(input_seq), torch.tensor(masked_target)))
        
    def __len__(self):
        return len(self.examples)
                    
    def __getitem__(self, idx) -> tuple:
        return self.examples[idx]
        
    def _truncate_or_pad(self, token_ids: list, target_len: int, pad_value: int) -> list:
        """Returns a truncated or padded version of the tokenized list, depending on its length
        """
        if len(token_ids) >= target_len:
            return token_ids[:target_len]
        else:
            return token_ids + [pad_value] * (target_len - len(token_ids))