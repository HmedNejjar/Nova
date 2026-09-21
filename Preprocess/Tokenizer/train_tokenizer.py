import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(1, str(ROOT))

from pathlib import Path
import yaml

from Preprocess.Tokenizer.tokenizer import BPE


def resolve_path(raw_path: str | Path) -> Path:
    """
    Resolve config paths relative to the project root.
    Absolute paths are returned as-is.
    """
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    return ROOT / path

def main() -> None:
    # Load tokenizer config
    config_path = ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)["Tokenizer"]
        
    vocab_size = int(config["vocab_size"])
    savepath = resolve_path(config["savepath"])
    corpus_path = resolve_path(config["corpus_path"])
    
    print("=== Tokenizer Training Config ===")
    print(f"  vocab_size:  {vocab_size}")
    print(f"  savepath:    {savepath}")
    print(f"  corpus_path: {corpus_path}")
    print("=" * 35)
    
    # Initialize and train the tokenizer
    bpe = BPE(vocab_size, savepath)
    bpe.train(corpus_path)
    
    print("\nTraining complete.")
    print(f"  Model saved to: {bpe.tokenizer_path}")
    print(f"  Final vocab size: {len(bpe.vocab)}")

if __name__ == "__main__":
    main()
