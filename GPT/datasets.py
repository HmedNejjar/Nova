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
    """
    Conversational SFT dataset for Nova.

    Expected conversation format:

        <bos>
        <|system|>
        ...
        <|user|>
        ...
        <|assistant|>
        ...
        <eos>

    Important:
        - Windows NEVER cross an eos -> bos boundary.
        - Loss is applied only to assistant tokens.
        - <|assistant|> itself is also trained.
        - EOS is trained when it terminates an assistant response.
    """

    def __init__(self, conversations: list[str], tokenizer, seq_len: int, stride_coeff: int,) -> None:
        super().__init__()

        if not conversations:
            raise ValueError("conversations cannot be empty")

        if seq_len <= 0:
            raise ValueError("seq_len must be greater than 0")

        if stride_coeff <= 0:
            raise ValueError("stride_coeff must be greater than 0")

        self.seq_len = seq_len
        self.stride = max(1, seq_len // stride_coeff)

        # --------------------------------------------------
        # Special tokens
        # --------------------------------------------------

        try:
            self.bos_id = tokenizer.vocab["<bos>"]
            self.eos_id = tokenizer.vocab["<eos>"]

            self.user_id = tokenizer.vocab["<|user|>"]
            self.assistant_id = tokenizer.vocab["<|assistant|>"]
            self.system_id = tokenizer.vocab["<|system|>"]

        except KeyError as e:
            raise KeyError(
                f"Missing required special token in tokenizer vocabulary: {e}"
            )

        # --------------------------------------------------
        # Every item stored here is already a valid window.
        #
        # Each element:
        #
        #     (input_ids, labels)
        #
        # This means __getitem__ becomes very cheap.
        # --------------------------------------------------

        self.samples: list[tuple[torch.Tensor, torch.Tensor]] = []

        for convo in tqdm(conversations, desc="Building chatbot dataset",):
            if not isinstance(convo, str) or not convo.strip():
                continue

            tokens = tokenizer.encode(convo)

            if len(tokens) < 2:
                continue

            # Ensure the conversation is actually bounded.
            if tokens[0] != self.bos_id:
                raise ValueError("Conversation does not start with <bos>")

            if tokens[-1] != self.eos_id:
                raise ValueError("Conversation does not end with <eos>")

            labels = self._build_labels(tokens)

            self._create_windows(tokens, labels)

    # ======================================================
    # LABEL MASKING
    # ======================================================

    def _build_labels(self, tokens: list[int]) -> list[int]:
        """
        Create assistant-only labels.
        """

        labels = [-100] * len(tokens)

        training_assistant = False

        for i, token in enumerate(tokens):

            # ------------------------------------------
            # System / user turn
            # ------------------------------------------

            if token == self.system_id or token == self.user_id:
                training_assistant = False
                continue

            # ------------------------------------------
            # Assistant turn
            # ------------------------------------------

            if token == self.assistant_id:
                training_assistant = True

                # Teach Nova to emit the assistant marker.
                labels[i] = token

                continue

            # ------------------------------------------
            # End of conversation
            # ------------------------------------------

            if token == self.eos_id:

                # EOS should be learned as the end of the
                # assistant response.
                if training_assistant:
                    labels[i] = token

                training_assistant = False

                continue

            # ------------------------------------------
            # Normal token inside assistant response
            # ------------------------------------------

            if training_assistant:
                labels[i] = token

        return labels

    # ======================================================
    # WINDOW CREATION
    # ======================================================

    def _create_windows(self, tokens: list[int], labels: list[int]) -> None:
        """
        Create fixed-length windows inside ONE conversation.

        No window can cross EOS.

        The final window is allowed to be shorter than seq_len
        only internally; it is padded to seq_len.
        """

        conversation_length = len(tokens)

        # --------------------------------------------------
        # Conversation shorter than seq_len
        #
        # We keep it instead of throwing it away.
        # --------------------------------------------------

        if conversation_length <= self.seq_len:

            input_ids = tokens[:-1]
            target_ids = labels[1:]

            input_ids, target_ids = self._pad_window(input_ids, target_ids)

            self.samples.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))

            return

        # --------------------------------------------------
        # Normal sliding windows
        # --------------------------------------------------

        start = 0

        while start < conversation_length - 1:

            input_end = min(start + self.seq_len, conversation_length - 1)

            target_end = input_end + 1

            input_ids = tokens[start:input_end]
            target_ids = labels[start + 1:target_end]

            # ------------------------------------------------
            # The target may contain EOS.
            #
            # Once EOS occurs, this window MUST end there.
            # Otherwise we'd train across the conversation
            # boundary.
            # ------------------------------------------------

            eos_relative_positions = [i for i, token in enumerate(input_ids) if token == self.eos_id]

            # EOS should normally be the last token of a
            # conversation, but this makes the logic robust.
            if eos_relative_positions:

                eos_pos = eos_relative_positions[0]

                input_ids = input_ids[:eos_pos]
                target_ids = target_ids[:eos_pos]

            # ------------------------------------------------
            # If there are no usable tokens, stop.
            # ------------------------------------------------

            if not input_ids:
                break

            # ------------------------------------------------
            # Check whether the TARGET contains EOS.
            #
            # If it does, we want that EOS to be the final
            # target token and then stop generating windows
            # for this conversation.
            # ------------------------------------------------

            contains_eos_target = self.eos_id in target_ids

            if contains_eos_target:

                eos_pos = target_ids.index(self.eos_id)

                input_ids = input_ids[:eos_pos + 1]
                target_ids = target_ids[:eos_pos + 1]

                input_ids, target_ids = self._pad_window(input_ids, target_ids)

                self.samples.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))

                break

            # ------------------------------------------------
            # Normal full window.
            # ------------------------------------------------

            if len(input_ids) == self.seq_len:
                self.samples.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))

            else:
                # Final partial window.
                input_ids, target_ids = self._pad_window(input_ids, target_ids,)

                self.samples.append((torch.tensor(input_ids, dtype=torch.long), torch.tensor(target_ids, dtype=torch.long)))
                break

            # ------------------------------------------------
            # Move forward.
            # ------------------------------------------------

            next_start = start + self.stride

            # Prevent infinite loops.
            if next_start <= start:
                break

            start = next_start

    # ======================================================
    # PADDING
    # ======================================================

    def _pad_window(
        self,
        input_ids: list[int],
        target_ids: list[int],
    ) -> tuple[list[int], list[int]]:
        """
        Pad the final window to seq_len.

        Input padding:
            BOS token is NOT used.
            We use EOS as the harmless padding token.

        Target padding:
            -100 so CrossEntropyLoss ignores it.
        """

        padding_length = self.seq_len - len(input_ids)

        if padding_length <= 0:
            return input_ids, target_ids

        input_ids = input_ids + [self.eos_id] * padding_length

        target_ids = target_ids + [-100] * padding_length

        return input_ids, target_ids

    # ======================================================
    # DATASET API
    # ======================================================

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int,) -> tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]