import numpy as np

class UniversalOptimizer:
    def __init__(self, params, lr=1e-3, method='adam', momentum=0.9, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = params  # Expects a list of Tensor objects
        self.lr = lr
        self.method = method.lower()
        self.momentum = momentum
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        
        # State storage for moments (velocity and squared gradients)
        self.state = [{'v': np.zeros_like(p.data), 's': np.zeros_like(p.data)} for p in params]

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        self.t += 1
        
        for i, param_tensor in enumerate(self.params):
            if param_tensor.grad is None:
                continue
            
            grad = param_tensor.grad
            param = param_tensor.data

            # 1. Apply Weight Decay (L2 Regularization)
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
            
            # Update the tensor's data
            param_tensor.data = param
