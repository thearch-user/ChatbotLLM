import torch
from torch.nn import functional as F

class SemiSupervisedTrainer:
    def __init__(self, model, optimizer, confidence_threshold=0.9):
        self.model = model
        self.optimizer = optimizer
        self.confidence_threshold = confidence_threshold

    def train_step(self, input_ids):
        logits = self.model(input_ids)
        probs = F.softmax(logits, dim=-1)

        max_probs, pseudo_labels = probs.max(dim=-1)
        mask = max_probs > self.confidence_threshold

        if mask.sum() == 0:
            return None  # skip low-confidence batch

        loss = F.cross_entropy(
            logits[mask],
            pseudo_labels[mask]
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
