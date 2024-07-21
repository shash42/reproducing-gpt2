from regextok import RegexTokenizer

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

tok = RegexTokenizer(GPT4_SPLIT_PATTERN)

#Tests for train
# print(tok.merge([5, 6, 6, 7, 9, 1], (6, 7), 99))

# with open("unicode.txt", "r", encoding="utf-8") as f:
#     text = f.read()
# tok.train(text, 276, verbose=True)

# import os
# with open("taylorswift.txt", "r", encoding="utf-8") as f:
#     text = f.read()
# os.makedirs("models", exist_ok=True)
# tok.train(text, 512, verbose=True)
# prefix = os.path.join("models", "regex_tokenizer.json")
# tok.save(prefix)

tok.load("models/regex_tokenizer.json.model")
ids = tok.encode("hello world!!!? (안녕하세요!) lol123 😉")
text = tok.decode(ids)
print(text)