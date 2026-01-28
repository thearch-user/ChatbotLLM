from core.tensor import Tensor
from core.nn.linear import Linear
from core.nn.attention import MultiHeadSelfAttention
from core.ops import layer_norm, relu, Dropout
import numpy as np

class TransformerEncoderBlock:
    def __init__(self, d_model, num_heads, dropout_rate=0.1):
        self.attention = MultiHeadSelfAttention(d_model, num_heads)
        self.norm1_gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.norm1_beta = Tensor(np.zeros(d_model), requires_grad=True)
        self.dropout1 = Dropout(dropout_rate)

        self.ffn_linear1 = Linear(d_model, 4 * d_model) # Typically 4*d_model for hidden dim
        self.ffn_linear2 = Linear(4 * d_model, d_model)
        self.norm2_gamma = Tensor(np.ones(d_model), requires_grad=True)
        self.norm2_beta = Tensor(np.zeros(d_model), requires_grad=True)
        self.dropout2 = Dropout(dropout_rate)

    def __call__(self, x: Tensor, mask=None, training: bool = True):
        # Multi-Head Self-Attention
        attn_output = self.attention(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        x = x + attn_output # Residual connection
        x = layer_norm(x, self.norm1_gamma, self.norm1_beta) # Layer Normalization

        # Feed-Forward Network
        ffn_output = self.ffn_linear1(x)
        ffn_output = relu(ffn_output)
        ffn_output = self.ffn_linear2(ffn_output)
        ffn_output = self.dropout2(ffn_output, training=training)
        x = x + ffn_output # Residual connection
        x = layer_norm(x, self.norm2_gamma, self.norm2_beta) # Layer Normalization

        return x

    def parameters(self):
        params = self.attention.parameters()
        params += [self.norm1_gamma, self.norm1_beta]
        params += self.ffn_linear1.parameters()
        params += self.ffn_linear2.parameters()
        params += [self.norm2_gamma, self.norm2_beta]
        return params
