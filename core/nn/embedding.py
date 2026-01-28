from core.tensor import Tensor
import numpy as np

class Embedding:
    def __init__(self, vocab_size, d_model):
        # Initialize weights with small random values
        self.W = Tensor(np.random.randn(vocab_size, d_model) * 0.02, requires_grad=True)

    def __call__(self, x: Tensor):
        # x is a tensor of indices
        # We need to select rows from self.W based on the indices in x
        # This is essentially a fancy indexing operation
        
        # Ensure x contains valid indices within the vocab size
        if np.any(x.data < 0) or np.any(x.data >= self.W.data.shape[0]):
            raise IndexError("Embedding index out of bounds.")
            
        output_data = self.W.data[x.data]
        out = Tensor(output_data, requires_grad=True)

        # Backward pass: need to accumulate gradients into the correct rows of W
        def _backward():
            if self.W.requires_grad:
                # Use a scatter-add operation to add gradients to the appropriate rows of W
                # This is complex with numpy. A simpler approach for autograd might be
                # to iterate or use a specialized function if numpy doesn't support it well.
                # For now, let's assume a basic gradient accumulation.
                # A proper implementation might require more sophisticated handling depending on how indices are structured.
                
                # Let's try a simplified approach:
                # Get the indices used for lookup
                indices = x.data.flatten()
                gradients = out.grad.reshape(-1, out.grad.shape[-1]) # Flatten batch/sequence dim
                
                # Accumulate gradients for each word index
                # This can be slow if done naively.
                # For performance, one would use sparse updates or vectorized operations if available.
                
                # Using numpy's `add.at` for scatter-add like behavior
                np.add.at(self.W.grad, indices, gradients)

        out._backward = _backward
        out._prev = [x] # The input tensor (indices)
        return out

    def parameters(self):
        return [self.W]
