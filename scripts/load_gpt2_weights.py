import torch
import numpy as np
import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.model import Transformer
from model.model_args import ModelArgs
from core.tensor import Tensor

def load_gpt2_from_hf(model_type='gpt2'):
    """
    Downloads GPT-2 weights from Hugging Face and maps them to our custom Transformer model.
    Supported types: 'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'
    """
    from transformers import GPT2LMHeadModel
    
    print(f"Loading weights for {model_type} from Hugging Face...")
    model_hf = GPT2LMHeadModel.from_pretrained(model_type)
    sd_hf = model_hf.state_dict()
    
    # Model configuration
    config = {
        'gpt2':         dict(n_layers=12, n_heads=12, d_model=768),  # 124M params
        'gpt2-medium':  dict(n_layers=24, n_heads=16, d_model=1024), # 350M params
        'gpt2-large':   dict(n_layers=36, n_heads=20, d_model=1280), # 774M params
        'gpt2-xl':      dict(n_layers=48, n_heads=25, d_model=1600), # 1558M params
    }[model_type]
    
    args = ModelArgs(
        vocab_size=50257,
        d_model=config['d_model'],
        max_seq_len=1024,
        n_heads=config['n_heads'],
        n_layers=config['n_layers']
    )
    
    # Initialize our model
    model = Transformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        max_seq_len=args.max_seq_len,
        n_heads=args.n_heads,
        n_layers=args.n_layers
    )
    
    # Mapping HF state dict to our custom Transformer attributes
    # Our Transformer Structure:
    # self.embed.W
    # self.blocks[i].attention.wq.W, .b
    # self.blocks[i].attention.wk.W, .b
    # self.blocks[i].attention.wv.W, .b
    # self.blocks[i].attention.wo.W, .b
    # self.blocks[i].norm1_gamma, .norm1_beta
    # self.blocks[i].ffn_linear1.W, .b
    # self.blocks[i].ffn_linear2.W, .b
    # self.blocks[i].norm2_gamma, .norm2_beta
    # self.final_ln_gamma, .final_ln_beta
    # self.head.W, .b

    # GPT-2 HF structure notes:
    # transformer.wte.weight -> embed.W
    # transformer.h.i.attn.c_attn.weight (is 3*d_model, d_model) -> need to split and transpose
    # transformer.h.i.ln_1.weight -> norm1_gamma
    
    print("Mapping weights...")
    
    # 1. Embeddings
    model.embed.W.data = sd_hf['transformer.wte.weight'].numpy()
    
    # 2. Blocks
    for i in range(args.n_layers):
        block_hf = f'transformer.h.{i}'
        
        # Layer Norms
        model.blocks[i].norm1_gamma.data = sd_hf[f'{block_hf}.ln_1.weight'].numpy()
        model.blocks[i].norm1_beta.data = sd_hf[f'{block_hf}.ln_1.bias'].numpy()
        model.blocks[i].norm2_gamma.data = sd_hf[f'{block_hf}.ln_2.weight'].numpy()
        model.blocks[i].norm2_beta.data = sd_hf[f'{block_hf}.ln_2.bias'].numpy()
        
        # Attention (GPT-2 uses Conv1D, we use Linear. GPT-2 weights are transposed)
        # c_attn weights are (d_model, 3*d_model) -> split into Q, K, V
        w_qkv = sd_hf[f'{block_hf}.attn.c_attn.weight'].numpy() # (768, 2304)
        b_qkv = sd_hf[f'{block_hf}.attn.c_attn.bias'].numpy()   # (2304,)
        
        q_w, k_w, v_w = np.split(w_qkv, 3, axis=-1)
        q_b, k_b, v_b = np.split(b_qkv, 3, axis=-1)
        
        model.blocks[i].attention.wq.W.data = q_w
        model.blocks[i].attention.wq.b.data = q_b
        model.blocks[i].attention.wk.W.data = k_w
        model.blocks[i].attention.wk.b.data = k_b
        model.blocks[i].attention.wv.W.data = v_w
        model.blocks[i].attention.wv.b.data = v_b
        
        # Projection
        model.blocks[i].attention.wo.W.data = sd_hf[f'{block_hf}.attn.c_proj.weight'].numpy()
        model.blocks[i].attention.wo.b.data = sd_hf[f'{block_hf}.attn.c_proj.bias'].numpy()
        
        # FFN
        model.blocks[i].ffn_linear1.W.data = sd_hf[f'{block_hf}.mlp.c_fc.weight'].numpy()
        model.blocks[i].ffn_linear1.b.data = sd_hf[f'{block_hf}.mlp.c_fc.bias'].numpy()
        model.blocks[i].ffn_linear2.W.data = sd_hf[f'{block_hf}.mlp.c_proj.weight'].numpy()
        model.blocks[i].ffn_linear2.b.data = sd_hf[f'{block_hf}.mlp.c_proj.bias'].numpy()

    # 3. Final LN
    model.final_ln_gamma.data = sd_hf['transformer.ln_f.weight'].numpy()
    model.final_ln_beta.data = sd_hf['transformer.ln_f.bias'].numpy()
    
    # 4. Head (GPT-2 ties wte and lm_head weights usually, but we check)
    if 'lm_head.weight' in sd_hf:
        model.head.W.data = sd_hf['lm_head.weight'].numpy().T # Linear layer expects (in, out)
    else:
        # Weight tying
        model.head.W.data = sd_hf['transformer.wte.weight'].numpy().T

    print(f"Successfully loaded {model_type} weights into custom Transformer model.")
    return model

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_type = sys.argv[1]
    else:
        model_type = 'gpt2'
    
    # This requires 'transformers' library
    try:
        model = load_gpt2_from_hf(model_type)
        # In a real scenario, we might save the custom model state:
        # np.savez("gpt2_custom.npz", ...) 
    except ImportError:
        print("Error: 'transformers' and 'torch' libraries are required to download and convert weights.")
        print("Run: pip install transformers torch")
