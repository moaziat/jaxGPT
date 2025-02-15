import jax
import jax.numpy as jnp
import jax.random as random
from flax import nnx 

#read the data  
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# string to integer and integer to string mappings 
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# encode and decode functions 
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])


data = jnp.array(encode(text), dtype=jnp.int32)

#prepare train and validation sets 
n = 0.9*len(data)
train_data = data[:int(n)]
val_data = data[int(n):] 

#batching 
random.PRNGKey(1337)
batch_size = 4
block_size = 8

x = train_data[:block_size]
y = train_data[1:block_size+1]
for t in range(block_size):
    context = x[:t+1]
    target = y[t]


def get_batch(split): 
    data = train_data if split == 'train' else val_data
    ix = random.randint(random.PRNGKey(1337), (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

xb, yb = get_batch('train')

for b in range(batch_size): # batch dimension
    for t in range(block_size): # time dimension
        context = xb[b, :t+1]
        target = yb[b,t]


# CrossEntropy loss implementation with nnx 
class CrossEntropyLoss(nnx.Module):
    def __call__(self, logits, labels):
        log_probs = jax.nn.log_softmax(logits)
        one_hot_labels = jax.nn.one_hot(labels, logits.shape[-1])
        return -jnp.mean(jnp.sum(one_hot_labels * log_probs, axis=-1))
    

def token_embedding_table(vocab_size): 
    
    return nnx.Embed(vocab_size, vocab_size, rngs =nnx.Rngs(0) ) 

def forward(idx, targets): 
    
    logits = token_embedding_table(vocab_size)(idx)
    if targets is None: 
        loss = None 

    else:
        
        B, T, C = logits.shape
        logits  = logits.reshape(B*T, C)
        targets  = targets.reshape(B*T)
        loss  = CrossEntropyLoss()(logits, targets)

    return logits, loss

logits, loss = forward(xb, yb)

print(loss)

def generate(idx, max_new_tokens):
    # idx is (B, T) array of indices in the current context
    key = jax.random.PRNGKey(0)  # initial key
    for _ in range(max_new_tokens):
        # get the predictions
        logits, loss = forward(idx, None)
        # focus only on the last time step
        logits = logits[:, -1] # becomes (B, C)
        # apply softmax to get probabilities
        probs = jax.nn.softmax(logits, axis=-1) # (B, C)
        # get a new key for each iteration
        key, subkey = jax.random.split(key)
        # sample from the distribution using the new key
        idx_next = jax.random.categorical(subkey, probs, axis=-1)[:, None] # (B, 1)
        # append sampled index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1) # (B, T+1)
    return idx

x = jnp.zeros((1, 1), dtype=jnp.int32)
print(decode(generate(x, 100)[0].tolist()))
