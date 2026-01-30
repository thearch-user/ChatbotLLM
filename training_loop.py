import os
import math
import random
import numpy as np
import pickle
from glob import glob
from typing import List

from model.model import Transformer
from core.tensor import Tensor
from core.ops import cross_entropy
from tokenizor import tokenizor_train

# ---------------------------
# Config
# ---------------------------

BATCH_SIZE = 4 # Reduced for CPU/Custom autograd performance
SEQ_LEN = 128  # Reduced for performance
LR = 3e-4
WEIGHT_DECAY = 0.1
EPOCHS = 1
GRAD_CLIP = 1.0
LOG_EVERY = 10
SAVE_EVERY = 100

DATASET_DIR = "datasets"
CHECKPOINT_DIR = "checkpoints"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# ---------------------------
# Dataset Loader
# ---------------------------

def load_all_texts() -> List[str]:
    files = glob(f"{DATASET_DIR}/**/*.txt", recursive=True)
    texts = []
    for f in files:
        with open(f, "r", encoding="utf-8", errors="ignore") as file:
            texts.append(file.read())
    return texts


def build_token_stream(tokenizer, texts):
    token_stream = []
    for text in texts:
        ids = tokenizer.encode(text)
        token_stream.extend(ids)
    return np.array(token_stream, dtype=np.int32)


def get_batch(token_stream):
    ix = np.random.randint(0, len(token_stream) - SEQ_LEN - 1, (BATCH_SIZE,))
    x_data = np.stack([token_stream[i:i+SEQ_LEN] for i in ix])
    y_data = np.stack([token_stream[i+1:i+SEQ_LEN+1] for i in ix])
    
    x = Tensor(x_data, requires_grad=False)
    y = Tensor(y_data, requires_grad=False)
    return x, y


# ---------------------------
# Training
# ---------------------------

def train():
    print("🔥 Loading tokenizer")
    tokenizer = tokenizor_train.load_tokenizer("tokenizer.json")

    print("📚 Loading datasets")
    texts = load_all_texts()
    if not texts:
        print("⚠️ No text files found in datasets directory. Creating dummy stream.")
        token_stream = np.random.randint(0, tokenizer.vocab_size, (10000,), dtype=np.int32)
    else:
        token_stream = build_token_stream(tokenizer, texts)

    print(f"🧮 Total tokens: {len(token_stream):,}")

    print("🧠 Initializing model")
    # Using smaller dimensions for custom autograd speed
    model = Transformer(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        max_seq_len=SEQ_LEN,
        n_heads=4,
        n_layers=4,
        dropout_rate=0.1
    )
    model.train()

    optimizer = model.get_optimizer(
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY
    )

    step = 0

    print("🚀 Starting training")

    for epoch in range(EPOCHS):
        print(f"\n🌍 Epoch {epoch+1}/{EPOCHS}")

        num_batches = len(token_stream) // (BATCH_SIZE * SEQ_LEN)
        for b in range(num_batches):
            x, y = get_batch(token_stream)

            optimizer.zero_grad()

            # Forward pass
            logits = model(x)
            
            # Loss calculation
            loss = cross_entropy(logits, y)

            # Backward pass
            loss.backward()

            # Gradient clipping (manual implementation)
            if GRAD_CLIP > 0:
                total_norm = 0
                params = model.parameters()
                for p in params:
                    if p.grad is not None:
                        total_norm += np.sum(p.grad**2)
                total_norm = np.sqrt(total_norm)
                clip_coef = GRAD_CLIP / (total_norm + 1e-6)
                if clip_coef < 1:
                    for p in params:
                        if p.grad is not None:
                            p.grad *= clip_coef

            # Optimizer step
            optimizer.step()

            if step % LOG_EVERY == 0:
                print(f"step {step:05d} | loss {loss.data:.4f}")

            if step % SAVE_EVERY == 0 and step > 0:
                ckpt_path = f"{CHECKPOINT_DIR}/model_step_{step}.pkl"
                # Save model parameters using pickle
                params_data = [p.data for p in model.parameters()]
                with open(ckpt_path, "wb") as f:
                    pickle.dump(params_data, f)
                print(f"💾 Saved checkpoint {ckpt_path}")

            step += 1

    print("✅ Training complete")


if __name__ == "__main__":
    train()