from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

class BPE:
    """
    Lossless byte-level BPE tokenizer.
    """
    
    # Define special tokens
    SPECIAL_TOKENS = ("<bos>", "<eos>", "<system>", "</system>", "<user>", "</user>", "<assistant>", "</assistant>", "<thinking>", "</thinking>", "<pad>", "<unk>")
    
    def __init__(self, vocab_size: int, savepath: str | Path) -> None:
        self.vocab_size = vocab_size
        self.savepath = Path(savepath)
        self.savepath.mkdir(parents= True, exist_ok= True)
        
        self.tokenizer_path = self.savepath / "tokenizer.json"
        
        # Initialize Rust tokenizer
        if self.tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        else:
            self.tokenizer = self._build_empty()
    
    @staticmethod   
    def _build_empty() -> Tokenizer:
        """
        Build an empty tokenizer with default settings.
        """
        tokenizer = Tokenizer(models.BPE(unk_token= "<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space= False, use_regex=True)
        tokenizer.decoder = decoders.ByteLevel()
        
        return tokenizer
        
            
    def train(self, corpus_path: str | Path) -> None:
        trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, special_tokens=self._special_tokens(), initial_alphabet=pre_tokenizers.ByteLevel().alphabet(), show_progress=True)
        
        # STREAMING: Yield lines one by one so RAM doesn't explode
        def file_iterator():
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield line
        
        # Train the tokenizer
        print(f"Start Training on corpus: {corpus_path}")
        self.tokenizer.train_from_iterator(file_iterator(), trainer=trainer)
        
        # Save the tokenizer to a JSON file
        self.tokenizer.save(str(self.tokenizer_path))
        print(f"Saved tokenizer to: {self.tokenizer_path}")
        
    def encode(self, text: str) -> list[int]:
        """
        Encode text into token IDs.
        """
        return self.tokenizer.encode(text).ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        """
        Decode token IDs back into text.

        Tokens are stored in byte-level form ("Ġ" = space, "Ċ" = newline,
        multi-byte UTF-8 split across symbols), so they can't just be joined:
        the tokenizer's ByteLevel decoder maps them back to the exact original
        text, whitespace included.
        """
        if not token_ids:
            return ""
        return self.tokenizer.decode([int(t) for t in token_ids], skip_special_tokens=skip_special_tokens)
    
    @property
    def vocab(self) -> dict[str, int]:
        """Compatibility property: returns {token: id} mapping."""
        return self.tokenizer.get_vocab()
    
    @property
    def merges(self) -> dict[tuple[str, str], str]:
        """
        Compatibility property: returns merge rules as {(left, right): merged}.
        
        Note: Accessing this converts Rust merges to a Python dict.
        Avoid calling during training on large vocabularies.
        """
        result: dict[tuple[str, str], str] = {}
        raw_merges = getattr(self.tokenizer.model, "merges", [])
        for pair in raw_merges:
            try:
                left, right = pair
                result[(left, right)] = f"{left}{right}"
            except Exception:
                continue
        return result
    
    def _special_tokens(self) -> list:
        """Compatibility method returning special tokens tuple."""
        return list(self.SPECIAL_TOKENS)