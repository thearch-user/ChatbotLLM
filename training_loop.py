import math
import torch
import random
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

from ChatbotLLM.model.model import Transformer
from ChatbotLLM.tokenizor import tokenizor


# =====================
# CONFIG
# =====================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
SEQ_LEN = 256
EPOCHS = 3
LR = 3e-4
GRAD_CLIP = 1.0

DATA_PATH = "datasets/wiki/wiki.txt"


# =====================
# SIMPLE DATASET
# =====================
class TextDataset(Dataset):
    def __init__(self, path, tokenizer, seq_len):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# =====================
# MINI-BATCH SGD (SIMPLE)
# =====================
class SGD:
    def __init__(self, params, lr):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad


# =====================
# TRAINING LOOP
# =====================
def train():
    dataset = TextDataset(
        path=DATA_PATH,
        tokenizer=tokenizor,
        seq_len=SEQ_LEN,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    model = Transformer().to(DEVICE)
    model.train()

    optimizer = SGD(model.parameters(), lr=LR)

    global_step = 0

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for step, (tokens, targets) in enumerate(dataloader):
            tokens = tokens.to(DEVICE)
            targets = targets.to(DEVICE)

            # ── Forward ───────────────────────
            logits = model(tokens)   # [B, T, V]

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
            )

            # ── Backward ──────────────────────
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                GRAD_CLIP,
            )

            optimizer.step()

            total_loss += loss.item()
            global_step += 1

            if step % 50 == 0:
                print(
                    f"epoch {epoch} | step {step} | "
                    f"loss {loss.item():.4f} | "
                    f"ppl {math.exp(loss.item()):.2f}"
                )

        avg_loss = total_loss / len(dataloader)
        print(
            f"\n✅ Epoch {epoch} DONE | "
            f"avg loss {avg_loss:.4f} | "
            f"ppl {math.exp(avg_loss):.2f}\n"
        )


if __name__ == "__main__":
    train()
