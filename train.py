import jax
import jax.numpy as jnp
import optax
import argparse
import flax.nnx as nnx
import jax.random as random
from jax.sharding import PositionalSharding
from jaxGPT import jaxGPTLM, get_batch, decode, stoi 
import matplotlib.pyplot as plt



#---- Hyper-params
parser = argparse.ArgumentParser("jaxGPT trainer")
parser.add_argument('--max_iters', type=int, default=200, help="Maximum training iterations")
parser.add_argument('--lr', type=float, default=1e-3, help="Model learning rate")

args = parser.parse_args()
max_iters = args.max_iters
lr = args.lr


key = random.PRNGKey(1337)
#---- Sharding (training on multi-GPUs)
devices = jax.devices()
num_devices = len(devices)

#print training infos
print(f'Available training devices: {devices}')
print(f'max_iters set to : {max_iters}')
print(f'Learning rate set to {lr}')


sharding = PositionalSharding(devices)

def shard_data(data): 
   return jax.device_put(data, sharding.replicate())

def shard_params(params): 
   return jax.tree_util.tree_map(lambda p: jax.device_put(p, sharding.replicate()), params)



#--- Model init
model = jaxGPTLM(rngs=nnx.Rngs(1337))
params = nnx.state(model, nnx.Param)
param_count = sum(x.size for x in jax.tree_util.tree_leaves(params)) / 1e6 
print(f'Number of parameters: {param_count}')

sharded_params = shard_params(params)

#--- Optimizer init
optimizer = optax.adamw(lr)
opt_state = optimizer.init(sharded_params)


#--- most important step (I was sleep deprived just to understand how to pass model's params to the loss fun)
@nnx.jit
def loss_fn(params, x, y):
    nnx.update(model, params) #this is a crucial line 
    _, loss = model(x, y)  
    return loss 


#--- Training
@jax.jit
def train_step(params, opt_state, x, y): 
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state, loss

lossi = []
stepi = []
for iter in range(max_iters): 
  key, subkey = random.split(key)

  x, y = get_batch(subkey, 'train')
  x_shard, y_shard = shard_data(x), shard_data(y)

  sharded_params, opt_state, loss = train_step(sharded_params, opt_state, x_shard, y_shard)
  lossi.append(loss)
  if iter % 100 == 0: 
    print(f'step: {iter} -> loss: {loss}')
  stepi.append(iter)
  
#---- updates model with post-training params
nnx.update(model, sharded_params)
#-------------------------

#generate example
''' 
(**) The model takes too long to generate output, it trains faster on multiple GPUs but takes
longer to generate on multiple devices.
Because we use a fraction of each gpu to do a small computation. 
I'll try to come up with a solution later.
'''
print("#---------------------------Generating")
start = jnp.array([[stoi['\n']]])  # Example starting token
generated = model.generate(start, max_new_tokens=500, key=random.PRNGKey(0))
print(decode(generated[0].tolist()))

#-- Plot loss

plt.plot(stepi, lossi)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.show()




