import dataclasses

@dataclasses.dataclass
class ModelArgs:
    vocab_size: int = 50257  # Example: for GPT-2
    d_model: int = 768      # Example: hidden size
    max_seq_len: int = 1024 # Example: max sequence length
    n_heads: int = 12       # Example: number of attention heads
    n_layers: int = 12      # Example: number of transformer layers