import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = []

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            if self.requires_grad:
                # Handle broadcasting by summing over broadcasted dimensions
                grad_self = out.grad
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad = (self.grad if self.grad is not None else 0) + grad_self
            
            if other.requires_grad:
                grad_other = out.grad
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad = (other.grad if other.grad is not None else 0) + grad_other

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, requires_grad=(self.requires_grad or other.requires_grad))

        def _backward():
            if self.requires_grad:
                grad_self = out.grad * other.data
                while grad_self.ndim > self.data.ndim:
                    grad_self = grad_self.sum(axis=0)
                for axis, size in enumerate(self.data.shape):
                    if size == 1:
                        grad_self = grad_self.sum(axis=axis, keepdims=True)
                self.grad = (self.grad if self.grad is not None else 0) + grad_self
            
            if other.requires_grad:
                grad_other = out.grad * self.data
                while grad_other.ndim > other.data.ndim:
                    grad_other = grad_other.sum(axis=0)
                for axis, size in enumerate(other.data.shape):
                    if size == 1:
                        grad_other = grad_other.sum(axis=axis, keepdims=True)
                other.grad = (other.grad if other.grad is not None else 0) + grad_other

        out._backward = _backward
        out._prev = [self, other]
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self.data**other, requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                self.grad = (self.grad if self.grad is not None else 0) + (other * self.data**(other-1)) * out.grad

        out._backward = _backward
        out._prev = [self]
        return out

    def __matmul__(self, other):
        from core.ops import matmul
        return matmul(self, other)

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
