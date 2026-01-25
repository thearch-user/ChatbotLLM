from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=["[UNK]", "[BOS]", "[EOS]"]
)

tokenizer.train(["datasets/wiki.txt"], trainer)
tokenizer.save("tokenizor/tokenizer.json")

print("Tokenizer trained & saved.")
