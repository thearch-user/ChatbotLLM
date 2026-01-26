import numpy as np
from core.tensor import Tensor

class SelfAttention:
    def __init__(self, d_model):
        self.scale = 1.0 / np.sqrt(d_model)

    def __call__(self, q, k, v):
        scores = (q @ k.T) * self.scale
        weights = scores.softmax()
        return weights @ v
