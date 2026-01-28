import torch
import torch.nn as nn
import pandas as pd
from model.model_args import ModelArgs

dataset1 = open("ChatbotLLM/datastes/wiki.txt")
dataset = open("ChatbotLLM/datasets")

class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        args = ModelArgs()

        self.embed = nn.Embedding(args.vocab_size, args.d_model)
        self.pos = nn.Embedding(args.max_seq_len, args.d_model)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=args.d_model,
                nhead=args.n_heads,
                batch_first=True
            )
            for _ in range(args.n_layers)
        ])

        self.ln = nn.LayerNorm(args.d_model)
        self.head = nn.Linear(args.d_model, args.vocab_size, bias=False)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)

        x = self.embed(x) + self.pos(pos)
        for block in self.blocks:
            x = block(x)

        return self.head(self.ln(x))
