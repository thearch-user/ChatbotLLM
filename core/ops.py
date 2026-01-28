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

def sigmoid(a: Tensor):
    out = Tensor(1 / (1 + np.exp(-a.data)), requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + out.data * (1 - out.data) * out.grad

    out._backward = _backward
    out._prev = [a]
    return out

def tanh(a: Tensor):
    out = Tensor(np.tanh(a.data), requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + (1 - out.data**2) * out.grad

    out._backward = _backward
    out._prev = [a]
    return out

def relu(a: Tensor):
    out = Tensor(np.maximum(0, a.data), requires_grad=True)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + (a.data > 0) * out.grad

    out._backward = _backward
    out._prev = [a]
    return out

def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5):
    mean = np.mean(x.data, axis=-1, keepdims=True)
    variance = np.var(x.data, axis=-1, keepdims=True)
    x_normalized = (x.data - mean) / np.sqrt(variance + eps)
    out = Tensor(gamma.data * x_normalized + beta.data, requires_grad=True)

    def _backward():
        if x.requires_grad:
            # Gradients for x, gamma, beta
            dx_normalized = out.grad * gamma.data
            dvariance = np.sum(dx_normalized * (x.data - mean) * -0.5 * np.power(variance + eps, -1.5), axis=-1, keepdims=True)
            dmean = np.sum(dx_normalized * (-1 / np.sqrt(variance + eps)), axis=-1, keepdims=True) + dvariance * np.mean(-2 * (x.data - mean), axis=-1, keepdims=True)
            dx = dx_normalized / np.sqrt(variance + eps) + dvariance * 2 * (x.data - mean) / len(x.data[0]) + dmean / len(x.data[0])

            x.grad = (x.grad if x.grad is not None else 0) + dx

        if gamma.requires_grad:
            dgamma = np.sum(out.grad * x_normalized, axis=0, keepdims=True)
            gamma.grad = (gamma.grad if gamma.grad is not None else 0) + dgamma

        if beta.requires_grad:
            dbeta = np.sum(out.grad, axis=0, keepdims=True)
            beta.grad = (beta.grad if beta.grad is not None else 0) + dbeta

    out._backward = _backward
    out._prev = [x, gamma, beta]
    return out

class Dropout:
    def __init__(self, p: float = 0.5):
        self.p = p
        self.mask = None

    def __call__(self, x: Tensor, training: bool = True):
        if not training or self.p == 0:
            self.mask = np.ones(x.data.shape)
            return x

        self.mask = (np.random.rand(*x.data.shape) > self.p) / (1 - self.p)
        out = Tensor(x.data * self.mask, requires_grad=True)

        def _backward():
            if x.requires_grad:
                x.grad = (x.grad if x.grad is not None else 0) + out.grad * self.mask

        out._backward = _backward
        out._prev = [x]
        return out

def cross_entropy(logits: Tensor, target: Tensor):
    # Logits: (B, T, V), Target: (B, T) - indices
    B, T, V = logits.data.shape
    logits_reshaped = logits.data.reshape(-1, V)
    target_reshaped = target.data.reshape(-1)
    
    # Softmax stable implementation
    max_logits = np.max(logits_reshaped, axis=1, keepdims=True)
    exp_logits = np.exp(logits_reshaped - max_logits)
    softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Negative log likelihood
    nll = -np.log(softmax_probs[np.arange(len(target_reshaped)), target_reshaped] + 1e-10)
    loss_data = np.mean(nll)
    
    out = Tensor(loss_data, requires_grad=True)

    def _backward():
        if logits.requires_grad:
            # Gradient of CE with respect to logits: (softmax(logits) - target) / batch_size
            d_logits = softmax_probs.copy()
            d_logits[np.arange(len(target_reshaped)), target_reshaped] -= 1
            d_logits /= len(target_reshaped)
            
            logits.grad = (logits.grad if logits.grad is not None else 0) + d_logits.reshape(B, T, V) * out.grad

    out._backward = _backward
    out._prev = [logits]
    return out
