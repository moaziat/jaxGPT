import jax
import optax
import jax.numpy as jnp
import jax.random as random
from flax import nnx 



#-- Hyperparameters 
batch_size = 64
block_size = 256
max_iter = 500
#-------------

random.PRNGKey(1337)


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




#prepare train and validation sets 
data = jnp.array(encode(text), dtype=jnp.int32)
n = 0.9*len(data)
train_data = data[:int(n)]
val_data = data[int(n):] 


#data batching 
def get_batch(key, split='train'): 
    data = train_data if split == 'train' else val_data
    ix = random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i:i+block_size] for i in ix])
    y = jnp.stack([data[i+1:i+block_size+1] for i in ix])
    return (x, y), key



class BigramLM(nnx.Module):
    def __init__(self, vocab_size):
        # Initialize embedding with proper scaling
        key = random.PRNGKey(1337)
        self.token_embedding = jnp.array(
            random.normal(key, (vocab_size, vocab_size)) * 0.02
        )

    def __call__(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        logits = self.token_embedding[idx]  # (B,T,C)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(-1, C)
            targets = targets.reshape(-1)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        return logits, loss

    def generate(self, idx, max_new_tokens, key):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, _ = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = jax.nn.softmax(logits, axis=-1)
            # sample from the distribution
            key, subkey = random.split(key)
            idx_next = random.categorical(subkey, probs, axis=-1)
            # append sampled index to the running sequence
            idx = jnp.concatenate((idx, idx_next[:, None]), axis=1)
        return idx

        



model = BigramLM(vocab_size)
optimizer = optax.adam(1e-3)
opt_state = optimizer.init(model.token_embedding)

@jax.jit
def train_step(parameters, opt_state, batch):

  def loss_fn(p):
    model.token_embedding = p
    _, loss = model(xb, yb)
    print(loss)
    return loss 

  loss, grads = jax.value_and_grad(loss_fn)(parameters)
  updates, new_opt_state = optimizer.update(grads, opt_state, parameters)
  new_params = optax.apply_updates(parameters, updates)

  return new_params, new_opt_state, loss

params = model.token_embedding
key = random.PRNGKey(1337)
for step in range(500):
    batch, key = get_batch(key, 'train')
    xb, yb = batch
    params, opt_state, loss = train_step(params, opt_state, batch)
    if step % 10 == 0:
        print(f'step {step}: loss {loss:.4f}')