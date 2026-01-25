import json
from collections import defaultdict

class ScratchBPETokenizer:
    def __init__(self, vocab_size=5000, special_tokens=None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        self.vocab = {}
        self.merges = []
        self.stoi = {}
        self.itos = {}

    # -------------------------------
    # Training
    # -------------------------------
    def train(self, text: str):
        # Initial tokenization (character-level + </w>)
        words = text.split()
        corpus = [list(word) + ["</w>"] for word in words]

        vocab = defaultdict(int)
        for word in corpus:
            vocab[tuple(word)] += 1

        def get_stats(vocab):
            pairs = defaultdict(int)
            for word, freq in vocab.items():
                for i in range(len(word)-1):
                    pairs[(word[i], word[i+1])] += freq
            return pairs

        def merge_pair(pair, vocab):
            new_vocab = {}
            a, b = pair
            for word, freq in vocab.items():
                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word)-1 and word[i] == a and word[i+1] == b:
                        new_word.append(a+b)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_vocab[tuple(new_word)] = freq
            return new_vocab

        # BPE loop
        while len(vocab) < self.vocab_size:
            stats = get_stats(vocab)
            if not stats:
                break
            best = max(stats, key=stats.get)
            vocab = merge_pair(best, vocab)
            self.merges.append(best)

        # Build vocab list
        tokens = set()
        for word in vocab:
            for token in word:
                tokens.add(token)

        full_vocab = self.special_tokens + sorted(list(tokens))
        self.vocab = full_vocab
        self.stoi = {t:i for i,t in enumerate(self.vocab)}
        self.itos = {i:t for t,i in self.stoi.items()}

    # -------------------------------
    # Encoding
    # -------------------------------
    def encode(self, text: str):
        words = text.split()
        tokens = []

        for word in words:
            chars = list(word) + ["</w>"]

            for a, b in self.merges:
                i = 0
                new = []
                while i < len(chars):
                    if i < len(chars)-1 and chars[i] == a and chars[i+1] == b:
                        new.append(a+b)
                        i += 2
                    else:
                        new.append(chars[i])
                        i += 1
                chars = new

            for tok in chars:
                tokens.append(self.stoi.get(tok, self.stoi["<UNK>"]))

        return tokens

    # -------------------------------
    # Decoding
    # -------------------------------
    def decode(self, ids):
        tokens = [self.itos[i] for i in ids]
        text = "".join(tokens).replace("</w>", " ")
        return text.strip()

    # -------------------------------
    # Save / Load
    # -------------------------------
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "vocab_size": self.vocab_size,
                "special_tokens": self.special_tokens,
                "vocab": self.vocab,
                "merges": self.merges
            }, f)

    @staticmethod
    def load(path):
        with open(path) as f:
            data = json.load(f)
        tok = ScratchBPETokenizer(data["vocab_size"], data["special_tokens"])
        tok.vocab = data["vocab"]
        tok.merges = [tuple(m) for m in data["merges"]]
        tok.stoi = {t:i for i,t in enumerate(tok.vocab)}
        tok.itos = {i:t for t,i in tok.stoi.items()}
        return tok
