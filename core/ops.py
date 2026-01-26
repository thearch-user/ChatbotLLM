from core.tensor import Tensor
import numpy as np

def matmul(a: Tensor, b: Tensor):
    out = Tensor(a.data @ b.data, requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + out.grad @ b.data.T
        if b.requires_grad:
            b.grad = (b.grad if b.grad is not None else 0) + a.data.T @ out.grad

    out._backward = _backward
    out._prev = [a, b]
    return out
