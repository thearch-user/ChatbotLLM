import torch
from torch.nn import functional as F

class RLTrainer:
    def __init__(self, model, optimizer, reward_fn):
        self.model = model
        self.optimizer = optimizer
        self.reward_fn = reward_fn

    def train_step(self, input_ids, attention_mask=None):
        """
        Basic REINFORCE-style update
        """
        logits = self.model(input_ids)
        probs = F.softmax(logits, dim=-1)

        sampled_tokens = torch.multinomial(
            probs.view(-1, probs.size(-1)), 1
        ).view(input_ids.shape)

        rewards = self.reward_fn(sampled_tokens)

        log_probs = torch.log(
            probs.gather(-1, sampled_tokens.unsqueeze(-1)).squeeze(-1)
        )

        loss = -(log_probs * rewards).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
