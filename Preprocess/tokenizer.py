import yaml
import json
from pathlib import Path

class BPE:
    def __init__(self, vocab_size: int, savepath: str | Path) -> None:
        self.vocab_size = vocab_size
        self.savepath = Path(savepath)

        # Store learned merge operations and the current vocabulary mapping.
        self.merges = self._load_merges()
        self.vocab = self._load_vocab()

    def _load_merges(self) -> dict:
        """Load merges from JSON file if it exists."""
        merges_path = self.savepath / "merges.json"
        if merges_path.exists():
            try:
                with open(merges_path, 'r', encoding='utf-8') as f:
                    merges = json.load(f)
                # Convert back to tuple keys if they were saved as strings
                # If saved as f"{k[0]}, {k[1]}", you need to parse them back
                # Or better, save with proper tuple keys
                parsed_merges = {}
                for key, value in merges.items():
                    # Parse "char1, char2" back to tuple
                    if isinstance(key, str) and ',' in key:
                        parts = key.split(', ')
                        parsed_merges[(parts[0], parts[1])] = value
                    else:
                        parsed_merges[tuple(key)] = value
                return parsed_merges
            except Exception as e:
                print(f"Error loading merges: {e}")
                return {}
        return {}

    def _load_vocab(self) -> dict:
        """Load vocab from JSON file if it exists."""
        vocab_path = self.savepath / "vocab.json"
        if vocab_path.exists():
            try:
                with open(vocab_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading vocab: {e}")
                return {}
        return {}
    
    def get_words_count(self, corpus: str) -> dict:
        """Count how often each word appears after splitting it into character tokens."""
        word_freq = {}
        words = corpus.split()

        for word in words:
            # Split each word into characters and append an end-of-word marker.
            word = list(word) + ["</w>"]
            word = tuple(word)
            word_freq[word] = word_freq.get(word, 0) + 1

        return word_freq

    def get_pair_count(self, word_freq: dict) -> dict:
        """Count how often each adjacent symbol pair appears in the word frequencies."""
        pair_freq = {}

        for word, freq in word_freq.items():
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freq[pair] = pair_freq.get(pair, 0) + freq

        return pair_freq

    def merge_pair(self, best_pair: tuple, word_freq: dict) -> dict:
        """Replace every occurrence of the best pair with a merged token."""
        new_token = "".join(best_pair)
        new_word_freq = {}

        for word, freq in word_freq.items():
            new_symbol, i = [], 0   #new_symbol is to re-represent the word after the merge of the best pair

            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                    new_symbol.append(new_token)
                    i += 2
                else:
                    new_symbol.append(word[i])
                    i += 1

            merged_symbol = tuple(new_symbol)
            new_word_freq[merged_symbol] = new_word_freq.get(merged_symbol, 0) + freq

        return new_word_freq
    
    def train(self, corpus: str) -> tuple:
        word_freq = self.get_words_count(corpus= corpus)
        
        # Start by getting speacial characters and individual characters as tokens
        special_tokens, alphabets = ("<pad>", "<unk>", "<bos>", "<eos>", "<sep>"), set()
        for word in word_freq.keys():
            alphabets.update(word)
            
        self.vocab = {tok: i for i, tok in enumerate(special_tokens + tuple(sorted(alphabets)))}

        print("starting training")
        j = 1
        while len(self.vocab) < self.vocab_size:
            
            pair_freq = self.get_pair_count(word_freq= word_freq)
            
            if not pair_freq: break # Early stopping due to no more pairs available
            
            best_pair = max(pair_freq, key= pair_freq.get) #type: ignore
            new_token = "".join(best_pair)
            word_freq = self.merge_pair(best_pair= best_pair, word_freq= word_freq)
            
            self.vocab[new_token] = len(self.vocab)
            self.merges[best_pair] = new_token       
            
            print(f"finished iteation {j}, vocab length: {len(self.vocab)}"); j+=1
            
        return (self.vocab, self.merges)
    
    def encode(self, text: str) -> list:
        """Encode the input text into a list of token IDs based on the learned vocabulary."""
        # ===== 1. GROUP CHARACTERS INTO AVAILABLE MERGES IN MEMORY =====
        words = text.split()
        token_ids = []
        
        for word in words:
            word_token = tuple(list(word) + ["</w>"])
            
            for pair, merged in self.merges.items():
                new_tokens, i = [], 0
                
                while i < len(word_token):
                    if i < len(word_token) - 1 and (word_token[i], word_token[i+1]) == pair:
                        new_tokens.append(merged)
                        i += 2
                    else:
                        new_tokens.append(word_token[i])
                        i += 1
                word_token = tuple(new_tokens)
                
        # ===== 2. CONVERT TOKENS INTO TOKEN IDS FROM self.vocab ===== 
            for token in word_token:
                token_ids.append(self.vocab.get(token, self.vocab.get("<unk>")))  # Use <unk> for unknown tokens
        
        return token_ids
    
    def decode(self, token_ids: list) -> str:
        """Decode a list of token IDs back into the original text."""
        # Reverse the vocabulary mapping to get tokens from IDs
        id_to_token = {id_: token for token, id_ in self.vocab.items()}
        
        # Convert token IDs back to tokens
        tokens = [id_to_token.get(id, "<unk>") for id in token_ids]
        
        # Reconstruct the original text from tokens
        text = "".join(tokens)
        
        # Remove the end-of-word markers and split into words
        text = text.replace("</w>", " ").strip()
        
        return str(text)
    

if __name__ == "__main__":
    CONFIG_FILEPATH = r"G:\Projects\Python\Nova\config.yaml"
    with open(CONFIG_FILEPATH, 'r') as f:
        config = yaml.safe_load(f)
        
        vocab_size=config["Tokenizer"]["vocab_size"]
        corpus_file = config["Tokenizer"]["corpus_file"]
        savepath = config["Tokenizer"]["savefile_path"]
        
        with open(corpus_file, 'r') as f:
            corpus = f.read()
        
    bpe = BPE(vocab_size= vocab_size, savepath= savepath)

    # Example usage
    text = "hi guys this is a Byte pair encoding implementation using python from scratch."
    token_ids = bpe.encode(text)
    print("Encoded token IDs:", token_ids)
    decoded_text = bpe.decode(token_ids)
    print("Decoded text:", decoded_text)
    
    
    