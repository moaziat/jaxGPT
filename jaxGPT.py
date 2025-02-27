import jax
import optax
import jax.numpy as jnp
import jax.random as random
from flax import nnx
from functools import partial
import matplotlib.pyplot as plt


#---------------------------------------------
# data loading
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
#----------------------------------------------


# Hyperparameters
batch_size = 16
block_size = 256
eval_interval = 100
eval_iters = 200
n_embd = 128 
n_head = 8
n_layer = 8
dropout = 0.0
tril_mask = jnp.tril(jnp.ones((block_size, block_size)))
key = random.PRNGKey(1337)


chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

data = jnp.array(encode(text), dtype=jnp.int32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

@partial(jax.jit, static_argnames=('split',))
def get_batch(key, split):
    data = train_data if split == 'train' else val_data
    ix = random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.take(data, jnp.arange(block_size)[None, :] + ix[:, None], axis=0)
    y = jnp.take(data, jnp.arange(block_size)[None, :] + ix[:, None] + 1, axis=0)
    return x, y

class Head(nnx.Module):
    def __init__(self, head_size, rngs):
        self.key = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.query = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.value = nnx.Linear(n_embd, head_size, use_bias=False, rngs=rngs)
        self.dropout = nnx.Dropout(dropout, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        wei = q @ jnp.swapaxes(k, -1, -2) * C**-0.5
        wei = jnp.where(tril_mask[:T, :T], wei, float('-inf'))
        wei = jax.nn.softmax(wei, axis=-1)
        wei = self.dropout(wei)
        return wei @ v

class MultiHeadAttention(nnx.Module):
    def __init__(self, num_heads, head_size, rngs):
        self.heads = [Head(head_size, rngs) for _ in range(num_heads)]
        self.proj = nnx.Linear(n_embd, n_embd, rngs=rngs)
        
    def __call__(self, x):
        out = jnp.concatenate([h(x) for h in self.heads], axis=-1)
        return self.proj(out)

class FeedForward(nnx.Module):
    def __init__(self, n_embd, rngs):
        self.net = nnx.Sequential(
            nnx.Linear(n_embd, 4 * n_embd, rngs=rngs),
            nnx.relu,
            nnx.Linear(4 * n_embd, n_embd, rngs=rngs)
        )
        
    def __call__(self, x):
        return self.net(x)

class Block(nnx.Module):
    def __init__(self, n_embd, n_head, rngs):
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, n_embd//n_head, rngs)
        self.ffwd = FeedForward(n_embd, rngs)
        self.ln1 = nnx.LayerNorm(n_embd, rngs=rngs)
        self.ln2 = nnx.LayerNorm(n_embd, rngs=rngs)
        
    def __call__(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

@jax.jit
def compute_loss(logits, targets): 
    return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()


class jaxGPTLM(nnx.Module):

    def __init__(self, rngs):
        self.tok_emb = nnx.Embed(vocab_size, n_embd, rngs=rngs)
        self.pos_emb = nnx.Embed(block_size, n_embd, rngs=rngs)
        self.blocks = nnx.Sequential(*[Block(n_embd, n_head, rngs) for _ in range(n_layer)])
        self.ln_f = nnx.LayerNorm(n_embd, rngs=rngs)
        self.lm_head = nnx.Linear(n_embd, vocab_size, rngs=rngs)

    def __call__(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.tok_emb(idx)
        pos_emb = self.pos_emb(jnp.arange(T))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B*T, C)
            targets = targets.reshape(B*T)
            loss = compute_loss(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens, key):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            key, subkey = random.split(key)
            idx_next = random.categorical(subkey, logits, axis=-1)
            idx = jnp.concatenate([idx, idx_next[:, None]], axis=-1)
        return idx    



