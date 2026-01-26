import torch

def length_reward(tokens, target_len=50):
    """
    Simple reward: closer to target length = higher reward
    """
    lengths = (tokens != 0).sum(dim=1)
    return -torch.abs(lengths - target_len).float()


def dummy_alignment_reward(tokens):
    """
    Placeholder: replace with real preference model later
    """
    return torch.ones(tokens.size(0))
