import sys
from pathlib import Path
ROOT = Path(__file__).parent.parent
sys.path.insert(1, str(ROOT))

import yaml
import pickle
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

from Nova import NovaLM
from Preprocess.tokenizer import BPE
from GPT.datasets import VocabDataset, ChatBotDataset

with open(ROOT / Path(r"config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

model_config = config["Model"]
train_config = config["Train"]
tokenizer_config = config["Tokenizer"]
datasets_config = config["Datasets"]

VOCAB_SIZE = tokenizer_config["vocab_size"]
SAVEPATH = tokenizer_config["savepath"]

VOCAB_TRAIN = ROOT / Path(datasets_config["SimpleStories_train"])
VOCAB_TEST = ROOT / Path(datasets_config["SimpleStories_test"])

NUM_LAYERS = model_config["num_layers"]
EMBED_DIM = model_config["embed_dim"]
NUM_HEADS = model_config["num_heads"]
MAX_SEQ_LEN = model_config["max_seq_len"]
STRIDE_COEFF = model_config["stride_coeff"]
LR = float(model_config["learning_rate"])
ROPE_BASE = model_config["rope_base"]

EPOCHS = train_config["epochs"]
BATCH_SIZE = train_config["batch_size"]

MODEL_SAVE_PATH = ROOT / Path(r'Model\\best_model.pth')

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Training on {DEVICE}")

def compute_accuracy(preds: Tensor, labels: Tensor) -> tuple:
    mask = (labels != -100)
    correct = ((preds == labels) & mask).sum().item()
    total = mask.sum().item()
    return correct, total

def evaluate(model: NovaLM, loss_fn: nn.CrossEntropyLoss, test_dl: DataLoader, epoch: int) -> tuple:
    model.eval()

    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0
    num_batches = 0

    pbar = tqdm(test_dl, desc=f"Eval Epoch {epoch}/{EPOCHS}")
    
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))

        preds = logits.argmax(dim=-1)
        correct, total = compute_accuracy(preds, y)

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total
        num_batches += 1

        pbar.set_postfix(loss=loss.item())

    eval_loss = total_loss / num_batches if num_batches else 0.0
    eval_accuracy = total_correct / total_tokens if total_tokens else 0.0
    return eval_loss, eval_accuracy

def train(model: NovaLM, optimizer: AdamW, loss_fn: nn.CrossEntropyLoss,scaler: GradScaler , dataloaders: tuple[DataLoader, DataLoader], epoch: int) -> tuple:
    model.train()
    train_dl, test_dl = dataloaders
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0.0
    num_batches = 0

    pbar = tqdm(train_dl, desc=f"Training Epoch {epoch}/{EPOCHS}")

    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast(enabled= (DEVICE == 'cuda')):
            logits = model(x)
            loss = loss_fn(logits.view(-1, VOCAB_SIZE), y.view(-1))
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        preds = logits.argmax(dim= -1)
        correct, total = compute_accuracy(preds, y)

        total_loss += loss.item()
        total_correct += correct
        total_tokens += total
        num_batches += 1

        pbar.set_postfix(loss=loss.item())

    train_loss = total_loss / num_batches if num_batches else 0.0
    train_accuracy = total_correct / total_tokens if total_tokens else 0.0
    
    eval_loss, eval_accuracy = evaluate(model, loss_fn, test_dl, epoch)

    return train_loss, train_accuracy, eval_loss, eval_accuracy



if __name__ == "__main__":
    Nova = NovaLM(vocab_size= VOCAB_SIZE, embed_dim= EMBED_DIM, num_layers= NUM_LAYERS, num_heads= NUM_HEADS, max_seq_len= MAX_SEQ_LEN, rope_base= ROPE_BASE).to(DEVICE)
    bpe_tokenizer = BPE(vocab_size= VOCAB_SIZE, savepath= SAVEPATH)

    with open(VOCAB_TRAIN, 'rb') as f:
        tokenized_vocab_train = pickle.load(f)

    with open(VOCAB_TEST, 'rb') as f:
        tokenized_vocab_test = pickle.load(f)

    vocab_train_ds = VocabDataset(tokenized_ds= tokenized_vocab_train, seq_len= MAX_SEQ_LEN, stride_coeff= STRIDE_COEFF)
    vocab_test_ds = VocabDataset(tokenized_ds= tokenized_vocab_test, seq_len= MAX_SEQ_LEN, stride_coeff= STRIDE_COEFF)


    vocab_train_dl = DataLoader(vocab_train_ds, batch_size= BATCH_SIZE, num_workers= 4)
    vocab_test_dl = DataLoader(vocab_test_ds, batch_size=BATCH_SIZE, num_workers= 4)

    print(f"Train batches: {len(vocab_train_dl)} | Test batches: {len(vocab_test_dl)}")

    loss_fn = nn.CrossEntropyLoss(ignore_index= -100)
    scaler = GradScaler()
    optimizer = AdamW(Nova.parameters(), lr = LR, weight_decay= 1e-6)
    
    train_accuracy_graph, train_loss_graph = [], []
    test_accuracy_graph, test_loss_graph = [], []
    
    for epoch in range(EPOCHS):
        train_loss, train_accuracy, eval_loss, eval_accuracy = train(Nova, optimizer, loss_fn, scaler, (vocab_train_dl, vocab_test_dl), epoch)
        
        print(f"Train loss: {train_loss:.3f} || Train Accuracy: {train_accuracy:.3f}")
        print(f"Eval loss: {eval_loss:.3f} || Eval Accuracy: {eval_accuracy:.3f}")
        
        train_accuracy_graph.append(train_accuracy)
        train_loss_graph.append(train_loss)
        
        test_accuracy_graph.append(eval_accuracy)
        test_loss_graph.append(eval_loss)
        
    # ---- Plot the loss and accuracy graphs ----
        
    epochs = list(range(EPOCHS))

    train_fig, train_ax = plt.subplots(figsize=(8, 4))
    train_ax.plot(epochs, train_accuracy_graph, color='orange', label='Accuracy')
    train_ax.plot(epochs, train_loss_graph, color='blue', label='Loss')
    train_ax.set_title('Training Metrics')
    train_ax.set_xlabel('Epoch')
    train_ax.set_ylabel('Value')
    train_ax.legend()
    train_ax.grid(True, alpha=0.3)
    train_fig.tight_layout()
    train_fig.savefig('train_metrics.png')
    plt.show()

    eval_fig, eval_ax = plt.subplots(figsize=(8, 4))
    eval_ax.plot(epochs, test_accuracy_graph, color='orange', label='Accuracy')
    eval_ax.plot(epochs, test_loss_graph, color='blue', label='Loss')
    eval_ax.set_title('Evaluation Metrics')
    eval_ax.set_xlabel('Epoch')
    eval_ax.set_ylabel('Value')
    eval_ax.legend()
    eval_ax.grid(True, alpha=0.3)
    eval_fig.tight_layout()
    eval_fig.savefig('eval_metrics.png')
    plt.show()

    print('Saved train_metrics.png and eval_metrics.png')


