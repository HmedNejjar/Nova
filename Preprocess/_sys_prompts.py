import random
import pickle
def add_system_prompts(conversations: list[str], system_prompts: list[str], fraction: float = 0.7, seed: int = 42) -> list[str]:
    rng = random.Random(seed)
    augmented = []

    for c in conversations:
        if rng.random() < fraction:
            prompt = rng.choice(system_prompts)
            # insert right after "<bos>", before "<|user|>"
            assert c.startswith("<bos>"), f"Unexpected format: {c[:30]}"
            rest = c[len("<bos>"):]   # everything after <bos>
            new_convo = f"<bos><|system|> {prompt} " + rest
        else:
            new_convo = c   # left unchanged — no system prompt

        augmented.append(new_convo)
    return augmented
if __name__ == "__main__":
    SYSTEM_PROMPTS = [
        "You are a helpful assistant.",
        "You are a concise assistant. Keep your answers short.",
        "You are a friendly assistant who explains things simply.",
        "You are a knowledgeable assistant. Provide accurate, factual answers.",
        "You are a polite and professional assistant.",
        "Respond clearly and directly to the user's question.",
        "You are an assistant that avoids giving personal opinions.",
        "You are a patient assistant who explains things step by step.",
    ]

    with open("Preprocess/Datasets/ultrachat_train.pkl", "rb") as f:
        train_conversations = pickle.load(f)
    with open("Preprocess/Datasets/ultrachat_test.pkl", "rb") as f:
        test_conversations = pickle.load(f)

    train_conversations = add_system_prompts(train_conversations, SYSTEM_PROMPTS, fraction=0.7)
    print("done")
    test_conversations = add_system_prompts(test_conversations, SYSTEM_PROMPTS, fraction=0.7)
    print("done")
    with open("Preprocess/Datasets/Ultrachat_train_system.pkl", "wb") as f:
        pickle.dump(train_conversations, f)
    with open("Preprocess/Datasets/Ultrachat_test_system.pkl", "wb") as f:
        pickle.dump(test_conversations, f)