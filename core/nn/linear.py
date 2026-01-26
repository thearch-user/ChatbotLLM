from core.tensor import Tensor
from core.ops import matmul
import numpy as np

class Linear:
    def __init__(self, in_dim, out_dim):
        self.W = Tensor(np.random.randn(in_dim, out_dim) * 0.02, True)
        self.b = Tensor(np.zeros(out_dim), True)

    def __call__(self, x):
        return matmul(x, self.W) + self.b
