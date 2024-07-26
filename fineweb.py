import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8) #100M tokens per shard, 100 shards

DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
print(len(fw))

enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']
def tokenize(doc):
    #tokenizes a single document and reutnrs numpy uint16 array
    tokens = [eot]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "dict size too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

nprocs = max(1, os.cpu_count()//2)
with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
    token_count = 0
    progress_bar = None
    #each process will tokenize 16 documents independently, will sync at writing tokens to shard
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        if token_count + len(tokens) < shard_size:
            #push tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Writing tokens to shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            #finish current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")  
            remainder = shard_size - token_count
            all_tokens_np[token_count:] = tokens[:remainder]
            progress_bar.update(remainder)
            write_datafile(filename, all_tokens_np)
            shard_index += 1
            progress_bar = None
            all_tokens_np[:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder
    if token_count > 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}.npy")
        write_datafile(filename, all_tokens_np[:token_count])