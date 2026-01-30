from core.tensor import Tensor
from core.nn.linear import Linear
from core.nn.embedding import Embedding
from core.nn.attention import MultiHeadSelfAttention
from core.ops import layer_norm, relu, Dropout
from core.nn.transformer_block import TransformerEncoderBlock
import numpy as np

# Placeholder for positional encoding using custom Tensor
def generate_positional_encoding(seq_len, d_model, device='cpu'):
    # This needs to be implemented using custom Tensor operations.
    # For now, create a numpy array and convert it to Tensor.
    # In a real implementation, this should be done directly with Tensor operations.
    
    position = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10000.0) / d_model))
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return Tensor(pe, requires_grad=False) # Positional encoding does not require gradients

class Transformer:
    def __init__(self, vocab_size, d_model, max_seq_len, n_heads, n_layers, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.training = True

        self.embed = Embedding(vocab_size, d_model)
        self.pos_encoding = generate_positional_encoding(max_seq_len, d_model)

        self.blocks = [
            TransformerEncoderBlock(d_model, n_heads, dropout_rate)
            for _ in range(n_layers)
        ]

        # The output layer needs to map from d_model to vocab_size
        # This should be a custom Linear layer
        self.head = Linear(d_model, vocab_size)

        # LayerNorm parameters for the final LayerNorm before the head
        # These are similar to the LayerNorms within blocks but for the final output
        self.final_ln_gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.final_ln_beta = Tensor(np.zeros(d_model), requires_grad=True)

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def __call__(self, x: Tensor, mask=None):
        B, T = x.data.shape
        
        # Ensure sequence length does not exceed max_seq_len
        if T > self.max_seq_len:
            raise ValueError(f"Input sequence length {T} exceeds max_seq_len {self.max_seq_len}")

        # 1. Embedding + Positional Encoding
        embedded = self.embed(x)
        
        # Slice positional encoding to match the sequence length of the input
        pos_embed = Tensor(self.pos_encoding.data[:T, :], requires_grad=False)
        
        x = embedded + pos_embed # Element-wise addition of Tensors

        # 2. Pass through Transformer Encoder Blocks
        # We need to apply each block sequentially. The blocks are Python objects, not nn.ModuleList.
        # They need to be called like functions.
        for block in self.blocks:
            x = block(x, mask=mask, training=self.training)

        # 3. Final Layer Normalization and Head
        x = layer_norm(x, self.final_ln_gamma, self.final_ln_beta)
        output = self.head(x)

        return output

    def parameters(self):
        params = self.embed.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += [self.final_ln_gamma, self.final_ln_beta]
        params += self.head.parameters()
        return params

    def get_optimizer(self, lr=1e-3, momentum=0.9, weight_decay=0.0):
        from algorithms.SGD import SGD
        return SGD(self.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
