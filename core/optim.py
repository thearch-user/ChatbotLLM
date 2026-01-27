import numpy as np

class UniversalOptimizer:
    def __init__(self, params, lr=1e-3, method='adam', momentum=0.9, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params  # Expects a list of numpy arrays
        self.lr = lr
        self.method = method.lower()
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # State storage for moments (velocity and squared gradients)
        self.state = [{'v': np.zeros_like(p), 's': np.zeros_like(p)} for p in params]

    def step(self, grads):
        """
        grads: a list of numpy arrays matching the shape of self.params
        """
        self.t += 1
        
        for i, (param, grad) in enumerate(zip(self.params, grads)):
            # 1. Apply Weight Decay (L2 Regularization)
            # This adds a penalty to the gradient based on the size of the weights
            if self.weight_decay > 0:
                grad = grad + self.weight_decay * param

            state = self.state[i]

            if self.method == 'sgd':
                # Momentum update: v = m*v + g
                state['v'] = self.momentum * state['v'] + grad
                # Param update: p = p - lr * v
                param -= self.lr * state['v']

            elif self.method == 'adam':
                b1, b2 = self.betas
                # Update biased first moment
                state['v'] = b1 * state['v'] + (1 - b1) * grad
                # Update biased second moment
                state['s'] = b2 * state['s'] + (1 - b2) * (grad ** 2)
                
                # Bias correction
                v_corr = state['v'] / (1 - b1 ** self.t)
                s_corr = state['s'] / (1 - b2 ** self.t)
                
                # Adaptive update
                param -= self.lr * v_corr / (np.sqrt(s_corr) + self.eps)

class Optimizer:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.params:
            p.grad = None

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        super().__init__(params, lr)

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            p.data -= self.lr * p.grad

import numpy as np
from core.tensor import Tensor
from core.nn.linear import Linear
from core.optim import SGD
from core.data import batch_iterator

# Fake dataset
X = np.random.randn(1024, 10)
Y = np.random.randn(1024, 1)

model = Linear(10, 1)
optimizer = SGD([model.W, model.b], lr=1e-2)

batch_size = 32
epochs = 10

for epoch in range(epochs):
    epoch_loss = 0.0

    for xb, yb in batch_iterator(X, Y, batch_size):
        x = Tensor(xb, requires_grad=False)
        y = Tensor(yb, requires_grad=False)

        pred = model(x)
        loss_val = ((pred.data - y.data) ** 2).mean()
        loss = Tensor(loss_val, requires_grad=True)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss_val

    print(f"Epoch {epoch}: loss={epoch_loss:.4f}")
