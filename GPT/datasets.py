import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

from tqdm import tqdm
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
    def __init__(self, conversations: list[str], tokenizer: BPE, seq_len: int, stride_coeff: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.stride = seq_len // stride_coeff 

        USER_ID = tokenizer.vocab["<|user|>"]
        ASSISTANT_ID = tokenizer.vocab["<|assistant|>"]

        all_tokens = []
        all_labels = []

        for convo in tqdm(conversations, desc="Tokenizing conversations"):
            tokens = tokenizer.encode(convo)

            # build masked labels for THIS conversation, same logic as before
            labels = tokens.copy()
            masking = True
            for i, token in enumerate(tokens):
                if token == USER_ID:
                    masking = True
                elif token == ASSISTANT_ID:
                    masking = False
                if masking:
                    labels[i] = -100

            all_tokens.extend(tokens)
            all_labels.extend(labels)

        self.tokenized_data = torch.tensor(all_tokens)
        self.label_data = torch.tensor(all_labels)

        self.start_indices = list(range(0, len(self.tokenized_data) - seq_len, self.stride))
        assert len(self.start_indices) > 0, "Not enough data for even one window"

    def __len__(self) -> int:
        return len(self.start_indices)

    def __getitem__(self, idx: int) -> tuple:
        start_idx = self.start_indices[idx]
        end_idx = start_idx + self.seq_len

        input_seq = self.tokenized_data[start_idx:end_idx]
        target_seq = self.label_data[start_idx + 1:end_idx + 1]

        return (input_seq, target_seq)