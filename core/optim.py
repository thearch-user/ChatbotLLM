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