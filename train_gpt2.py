from dataclasses import dataclass
import inspect
import torch
import math
import os
import time
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head, self.n_embd = config.n_head, config.n_embd
        self.c_attn = nn.Linear(self.n_embd, 3*self.n_embd) #combines qkv
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() #batches, sequence length, channels
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2) #splitting into 3 at the channels
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) #(B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1))) # is now (B, nh, T, T)
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        #contigous makes the order of elements in memory same as if tensor was made from scratch
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') #approx only used for historical purposes of reproducing GPT-2, just using gelu works to remove flat relu gradient 0 problem, and modern LLMs use swiglu etc.
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()

        #implementing pre-normalization, not normalizing residual stream to keep it clean
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024 #context length
    vocab_size: int = 50257 #50,000 BPE merges + 256 leaves + endoftext
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        #weight sharing saving 30% params - LMhead (768 -> 50257, similar tokens have similar probability) behaves similarly to token embeddings (50257 -> 768, similar tokens have similar embedding)
        self.transformer.wte.weight = self.lm_head.weight #makes the memory common, so now they're refering to same tensor.

        #init params
        self.apply(self._init_weights) #apply is a method of nn.module which iterates over modules and applies function passed

    def _init_weights(self, module):
        #Use GPT-2 initializations, layernorm is same as pytorch default so we dont change
        #the 0.02 value is close to the value of xavier initialization, which is 1/sqrt(num_features) = 1/sqrt(768) ~ 0.03
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'): #We do this as residual stream variance will increase, so we normalize by 1/sqrt(num_layers)
                std *= (2 * self.config.n_layer) ** -0.5 #here we do 2* because we have both attention block and mlp block applying in each 'layer'
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Context length is only {self.config.block_size}, cannot forward length {T}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) #positional indices
        pos_emb = self.transformer.wpe(pos) #positional embeddings (T, n_embd)
        tok_emb = self.transformer.wte(idx) #token embeddings (B, T, n_embd)
        x = tok_emb + pos_emb
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) #(B, T, vocab_size), each position has output of predicted next token. inefficient sampling
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #logits: (B, T, vocab_size) -> (B*T, vocab_size) and targets (B, T) -> (B*T) 
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizers(self, weight_decay, learning_rate, device):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nondecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        #we dont want to decay bias, layernorm etc, i.e. dim1 tensors
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nondecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nondecay_params = sum(p.numel() for p in nondecay_params)
        if master_process:
            print(f"num decayed paramter tensors: {len(decay_params)}, with {num_decay_params}")
            print(f"num nondecayed paramter tensors: {len(nondecay_params)}, with {num_nondecay_params}")
        
        #fuse optimizer updates
        fused_available = 'fused' in inspect.signature(torch.optim.Adam).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused Adam: {use_fused}")
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        
        return optimizer

import tiktoken
import numpy as np

def load_tokens(filename):
    npt = np.load(filename)
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B, self.T = B, T
        self.process_rank, self.num_processes = process_rank, num_processes
        #load tokens at init
        assert split in {'train', 'val'}
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"

        if master_process:
            print(f"loaded {len(shards)} shards for split {split}")

        #state
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B*T+1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        self.current_position += B*T*self.num_processes
        if self.current_position + B*T*self.num_processes + 1 > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = self.B * self.T * self.process_rank
        return x, y

def get_lr(step):
    if step < warmup_steps:
        return (step+1) / max_steps * max_lr #step+1 so we don't start with lr 0 at step 0
    elif step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#################################################################
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

ddp = int(os.environ.get('RANK', -1)) != -1 #check if its a ddp run
if ddp:
    assert torch.cuda.is_available()
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK']) #rank across nodes
    ddp_local_rank = int(os.environ['LOCAL_RANK']) #rank within a node
    ddp_world_size = int(os.environ['WORLD_SIZE']) #total number of processes
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif hasattr(torch.backends, "mps") and torch.backends.mps_is_available():
        device = 'mps'
    print(f'using device: {device}')

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

#get a data batch

total_batch_size = 2**19 #524,288 tokens
B = 32 #micro-batch size
T = 1024 #sequence length
assert(total_batch_size % (B*T) == 0)
grad_accum_steps = total_batch_size // (B*T*ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

print(f"I am GPU {ddp_rank}")

train_loader = DataLoaderLite(B, T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")

torch.set_float32_matmul_precision('high')
# this makes matmuls 8x faster, which overall led to a 2x speedup (as other individual floats are still fp32, and we get channel/memory bound)

model = GPT(GPTConfig(vocab_size=50304)) #this vocab_size adds extra unused (tiktoken has only 50257) tokens, but makes it 393*2^7 which improves matmulspeed.
model.to(device)
# model = torch.compile(model)
'''
compiles the model code, to remove python interpreter overhead when forward passing model 
and doing compiler optimizations since whole code can be known. 
Also optimizes GPU read/writes as memory <> gpu round trips are quite time-taking for variables.
Note:
CPU memory - large (say 1Tb), slow to access
GPU memory - smaller (say 40 gigs), faster to access
GPU on-chip - tiny (say 20mb), extremely fast to access
-- commented for now because pytorch 2.4 and earlier dont support compile with python 3.12
'''
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
    #handles parallelism of backward passing, averaging gradients with Reduce op, and dispatching updated to all processes 
raw_model = model.module if ddp else model

max_lr = 3e-4
min_lr = max_lr * 0.1
warmup_steps = 715 #375M (from gpt-3) / 2**19
max_steps = 19073 #10B // 2**19

# optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95))
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

for step in range(max_steps):
    t0 = time.time()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            #only surround forward and loss acc to pytorch documentation. Goes to bfloat16 (exp8, mant7)
            #autocast doesnt convert all functions to bfloat16, eg loss func calcs are precision sensitive so remain float32
            logits, loss = model(x, y)
            # import code; code.interact(local=locals()) #use to see dtype of logits,loss as a debugger
        loss = loss / grad_accum_steps #scale the loss due to grad_accum and it would be divided more if batch size was larger due to reduction=mean
        loss_accum += loss.detach()
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1) #sync gradients across processes, need only on last step before backward
        loss.backward() # adds to the existing gradients (+=), so make sure to zero_grad before this
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) #average loss acros batches in diff processes

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) #gradient clipping, makes sure gradients don't spike/add instability
    
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step() # updates paramters
    
    if device=='cuda': torch.cuda.synchronize() #wait for the gpu to finish the above work before computing time on cpu below
    t1 = time.time()
    dt = (t1 - t0)*1000 #difference in milliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size) / (t1 - t0)
    if master_process:
        print(f"step {step}, loss: {loss_accum.item():.4f}, norm: {norm:.4f}, lr: {lr:.4e}, dt: {dt:.2f}ms, tok/sec: {tokens_per_sec:.2f}") # .item gets the value out (on cpu) from loss tensor (on gpu)

if ddp:
    destroy_process_group()

import sys; sys.exit(0)
#torchrun --nproc_per_node=4 train_gpt2.py

# model = GPT.from_pretrained('gpt2')
# model.eval()
# model.to(device)
# print('loaded weights')

# num_return_sequences = 5
# max_length = 30
# enc = tiktoken.get_encoding('gpt2')
# tokens = enc.encode("Hello, I'm a language model")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1) #(8,) -> (1, 8) -> (5, 8)
# x = tokens.to(device)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x) #(5, T, vocab_size)
        logits = logits[:, -1, :] #we only care about prediction at last position
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
        ix = torch.multinomial(topk_probs, 1) # (B, 1)
        xcol = torch.gather(topk_indices, -1, ix) #get topk_indices[:}[ix]
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

'''
Train LOSSES
Random - 10.82
Initialization - 10.94
50 batch optimization - 6.47
'''

'''
Optimization tok/sec throughputs
Initial (batch size 16) - 19k
With set_float32_matmul_precision - 29k
With autocast - 39k
With torch.compile(model) - couldnt run
With flash attention - 100k :0
vocab size = 2^7 * 393 (50304) - 107k
fused adamW - 108k
'''

'''
Hyperparam Tuning Train losses
beta 0.9, 0.95 + gradient clipping = 6.0
cosinelr schedule + weight decay 0.1 = 6.0
gradient accumalation
'''

'''
STEPS 
--- Model implementation
implemented GPT module
implemented Block incl MLP, SelfAttn
implemented forward pass 
sampling from logits
--- Training
convert text to batches of input, targets
implement loss calculation
overfit to 1 batch
make a dataloader to optimize over batches
--- Efficiency Improvements
Weight sharing LMhead and token embedder
Better initialization
Lower precision training (set matmul precision, autocast)
Flash attention
Nice numbers (maximize 2^x factor)
--- Hyperparams
Optimizer betas
Gradient clipping
CosineLR with linear warmup
--- Distributed Training
DDP
Tokenize data in parallel, sync and store in shards for better disk usage

'''

'''
Important ideas
1. Lower training precision helps optimize tensorflops, gpu memory limits, and gpu memory bandwidth. Latter 2 are particularly important because even in well-tuned applications
we only end up utilizing tensor cores only 60%, rest time goes in data-fetching (memory bandwidth issue). Being able to store more in gpu memory means lesser fetches, and also
more data/params can be fetched per second at lower precision.
2. We can use even lower int8 precision in inference. We can't use for training because it is uniformly spaced which makes it difficult to model the normal distributions we want in training.
3. Matrix multiplies are broken down into 4x4 multiplies at the low tensor-core level.
4. TF32 just truncates the last 13 bits (out of 23) of mantissa precision, which makes matrix multiplies 8x faster. Cool thing is that the accumulator gives a fp32 output
so nothing seems to have changed at the high-level, just internally the multiply is happening with tf32, loosing some precision which doesnt hurt DL
5. Flash attention optimizes GPU read/write for attention matrix computation
6. Gradient accumulation - Hyperparam values depend on batch size, and if we can't fit a large batch in memory, we can simulate it sequentially over multiple micro-batches
'''