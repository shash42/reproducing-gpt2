from karpathybase import Tokenizer, replace_control_characters, render_token
from basic import BasicTokenizer
import regex as re

class RegexTokenizer(BasicTokenizer):
    def __init__(self, pattern):
        super().__init__()
        self.pattern = pattern

    def count_pairs(self, ids, pair_counts):
        for pair in zip(ids, ids[1:]):
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return pair_counts
    
    def regex_encode(self, text):
        chunks = re.findall(self.pattern, text)
        tokens = [list(chunk.encode("utf-8")) for chunk in chunks]
        return tokens
    
    def train(self, text, vocab_size, verbose=False):
        #split text based on regex patter in self.pattern
        tokens = self.regex_encode(text)    
        num_merges = vocab_size - 256
        if verbose:
            print(f"Starting length: {sum([len(chunk) for chunk in tokens])}, training {num_merges} merges")
        for i in range(num_merges):
            pair_counts = {}
            for chunk in tokens:
                pair_counts = self.count_pairs(chunk, pair_counts)
            
            best_pair = max(pair_counts, key=pair_counts.get)
            self.merges[best_pair] = 256+i
            for idx in range(len(tokens)):
                tokens[idx] = self.merge(tokens[idx], best_pair, 256+i)

            if verbose:
                print(f"{best_pair} -> {256+i}, curr_length: {sum([len(chunk) for chunk in tokens])}")

        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        for pair, to in self.merges.items():
            print(to, pair[0], pair[1])
            self.vocab[to] = self.vocab[pair[0]] + self.vocab[pair[1]]

        return

    def encode(self, text):
        tokens = self.regex_encode(text)
        while True:
            pair_counts = {}
            for chunk in tokens:
                if len(chunk) >= 2:
                    pair_counts = self.count_pairs(chunk, pair_counts)
            best_pair = min(self.merges, key=lambda x: pair_counts.get(x, float("inf")))
            if best_pair not in pair_counts:
                break

            for idx in range(len(tokens)):
                tokens[idx] = self.merge(tokens[idx], best_pair, self.merges[best_pair])
        
        ids = []
        for chunk in tokens:
            ids.extend(chunk)
        return ids

    def decode(self, ids):
        # expand ids, convert to bytes, and then to utf-8
        return b"".join([self.vocab[idx] for idx in ids]).decode("utf-8", errors="replace")
         