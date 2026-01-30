import numpy as np

class SGD:
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0):
        """
        Mini-batch Stochastic Gradient Descent optimizer.
        
        Args:
            params: Iterable of parameters to optimize (objects with .data and .grad attributes).
            lr: Learning rate.
            momentum: Momentum factor.
            weight_decay: Weight decay (L2 penalty).
        """
        self.params = list(params)
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocities = [np.zeros_like(p.data) for p in self.params]

    def zero_grad(self):
        """Sets the gradients of all optimized parameters to zero."""
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def step(self):
        """Performs a single optimization step."""
        for i, p in enumerate(self.params):
            if p.grad is not None:
                grad = p.grad
                if self.weight_decay > 0:
                    grad = grad + self.weight_decay * p.data

                if self.momentum > 0:
                    self.velocities[i] = self.momentum * self.velocities[i] + grad
                    update = self.velocities[i]
                else:
                    update = grad
                
                p.data -= self.lr * update
