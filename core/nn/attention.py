import numpy as np
from core.tensor import Tensor
from core.nn.linear import Linear
from core.ops import matmul

class MultiHeadSelfAttention:
    def __init__(self, d_model, num_heads):
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)

        self.wo = Linear(d_model, d_model)

        self.scale = 1.0 / np.sqrt(self.head_dim)

    def __call__(self, q, k, v, mask=None):
        B, T, C = q.data.shape # Batch, Sequence Length, C = d_model

        # 1. Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # 2. Split into heads and reshape
        q_heads = Tensor(q.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3), requires_grad=True)
        k_heads = Tensor(k.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3), requires_grad=True)
        v_heads = Tensor(v.data.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3), requires_grad=True)

        # 3. Scaled Dot-Product Attention for each head
        # (B, num_heads, T, head_dim) @ (B, num_heads, head_dim, T) -> (B, num_heads, T, T)
        scores = matmul(q_heads, Tensor(k_heads.data.transpose(0, 1, 3, 2), requires_grad=True)) * self.scale

        if mask is not None:
            # Apply mask (e.g., for decoder self-attention)
            # Ensure mask is broadcastable to scores shape
            scores.data[mask == 0] = -np.inf # assuming 0 means masked

        weights = softmax(scores) # Softmax on the last dimension (T)

        attention_output = matmul(weights, v_heads)

        # 4. Concatenate heads and reshape back
        attention_output = Tensor(attention_output.data.transpose(0, 2, 1, 3).reshape(B, T, C), requires_grad=True)

        # 5. Final linear projection
        output = self.wo(attention_output)
        return output

    def parameters(self):
        return self.wq.parameters() + self.wk.parameters() + self.wv.parameters() + self.wo.parameters()

def softmax(x: Tensor):
    # A simple softmax implementation, assumes last axis for reduction
    exp_x = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
    sum_exp_x = np.sum(exp_x, axis=-1, keepdims=True)
    out_data = exp_x / sum_exp_x
    out = Tensor(out_data, requires_grad=True)

    def _backward():
        if x.requires_grad:
            # Simplified for now, real softmax backward is more complex
            # This is a placeholder, a full backward needs the Jacobian
            # For self-attention, the gradient usually comes from the subsequent matmul
            # For simple cases: dx = out.grad * out.data * (1 - out.data)
            # More generally: dx = out @ (diag(out) - out.T) * dout

            # Here, we can pass the gradient directly through if we assume it's part of a larger graph
            # where the subsequent matmul's backward will handle the dependency.
            # For correctness, this needs to be a proper Jacobian product.
            # For now, let's use a simplified approach that works for typical usage patterns.
            # A more robust solution would involve explicit Jacobian calculation.
            # As a temporary simplification, we'll propagate a scaled gradient.
            grad_output_reshaped = out.grad[..., None] # Add a new dimension for broadcasting
            softmax_jacobian_product = out.data[..., None] * (np.eye(out.data.shape[-1]) - out.data[..., None, :]) # This needs careful broadcasting
            # This Jacobian calculation is complex for N-D arrays. 
            # For simplicity in this context and assuming it's part of a larger graph where gradients are handled implicitly by matmul,
            # a more practical approach for custom autograd often involves breaking it down or relying on the chain rule more directly.
            
            # For a more robust softmax backward:
            s = out.data
            grad_input = s * out.grad
            sum_grad_input = np.sum(grad_input, axis=-1, keepdims=True)
            grad_input -= s * sum_grad_input
            
            x.grad = (x.grad if x.grad is not None else 0) + grad_input

    out._backward = _backward
    out._prev = [x]
    return out
