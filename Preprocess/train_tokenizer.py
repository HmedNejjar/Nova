from tokenizer import BPE
from pathlib import Path
import yaml
import json

PARENT_FOLDER = Path(r"G:\\Projects\\Python\\Nova")

def main():
    # Load the configuration from a YAML file
    with open(PARENT_FOLDER / Path("config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    vocab_size = config["Tokenizer"]["vocab_size"]
    savepath = Path(config["Tokenizer"]["savepath"])
    corpus_path = Path(config["Tokenizer"]["corpus_path"])

    # Read the corpus from the specified file
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    # Initialize and train the BPE tokenizer
    bpe_tokenizer = BPE(vocab_size=vocab_size, savepath=savepath)
    vocab, merges = bpe_tokenizer.train(corpus=corpus)
    
    merges = {f"{k[0]}, {k[1]}": v for k,v in bpe_tokenizer.merges.items()}
    
    # Save the vocabulary and merges to a JSON file
    with open(PARENT_FOLDER / f'{savepath}\\vocab.json', 'w') as f:
        json.dump(vocab, f, indent=4)
    
    with open(PARENT_FOLDER / f'{savepath}\\merges.json', 'w') as f:
        json.dump(merges, f, indent=4)
    

if __name__ == "__main__":
    main()
