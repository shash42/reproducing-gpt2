from karpathybase import Tokenizer

class BasicTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()
    
    def count_pairs(self, ids):
        pair_counts = {}
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts
    
    def merge(self, tokens, best_pair, idx):
        new_ids = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == best_pair:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(tokens[i])
                i += 1
        return new_ids

    def train(self, text, vocab_size, verbose=False):
        tokens = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        if verbose:
            print(f"Starting length: {len(tokens)}, training {num_merges} merges")
        for i in range(num_merges):
            pair_counts = self.count_pairs(tokens)
            best_pair = max(pair_counts, key=pair_counts.get)
            tokens = self.merge(tokens, best_pair, 256+i)
            self.merges[best_pair] = 256+i
            if verbose:
                print(f"{best_pair} -> {256+i}, curr_length: {len(tokens)}")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, to in self.merges.items():
            self.vocab[to] = self.vocab[pair[0]] + self.vocab[pair[1]]

        return

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            pair_counts = self.count_pairs(tokens)
            best_pair = min(self.merges, key=lambda x: pair_counts.get(x, float("inf")))
            if best_pair not in pair_counts:
                break
            tokens = self.merge(tokens, best_pair, self.merges[best_pair])
        return tokens

    def decode(self, ids):
        # expand ids, convert to bytes, and then to utf-8
        return b"".join([self.vocab[idx] for idx in ids]).decode("utf-8", errors="replace")
         


