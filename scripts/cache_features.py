#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
from saex.iterable_dataset import IterableDatasetConfig
from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.haver import ModelHaver, SAEHaver
from saex.sae import SAEConfig
from more_itertools import chunked


n_features = 3072
batch_size = 64
layer = 20
dataset_config = IterableDatasetConfig(
    dataset_name="nev/openhermes-2.5-phi-format-text",
    # dataset_name="nev/generated-phi-format-text",
)
model_config = MicrlhfModelConfig(
    tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
    gguf_path="../weights/phi-3-16.gguf",
    device_map="auto",
    use_flash=False,
    layer=layer,
    max_seq_len=128,
)
sae_config = SAEConfig(
    n_dimensions=n_features,
    batch_size=batch_size,
    expansion_factor=32,
    use_encoder_bias=True,
    remove_decoder_bias=False,
    encoder_init_method="orthogonal",
    decoder_init_method="pseudoinverse",
    decoder_bias_init_method="zeros",
    is_gated=False,
)
haver = ModelHaver(model_config=model_config, dataset_config=dataset_config)


# In[2]:

from micrlhf.utils.load_sae import get_sae
import os
os.makedirs("weights/sae", exist_ok=True)
sae_path = get_sae(layer, 8, model_dir="weights", return_fname=True)
haver_sae = SAEHaver(
    sae_config=sae_config,
    mesh=haver.mesh,
    sae_restore=sae_path)


# In[64]:

import pyarrow as pa
from tqdm.auto import trange
import pyarrow.parquet as pq

n_batches = 100
token_batches = []
for _, texts in zip(trange(n_batches), chunked(haver.create_dataset(), batch_size)):
    tokens = haver.model.to_tokens(texts)
    token_batches.append(pa.RecordBatch.from_arrays([tokens.reshape(-1)], ["tokens"]))
token_table = pa.Table.from_batches(token_batches)
pq.write_table(token_table, "weights/tokens.parquet",
               compression="snappy")

# In[63]:


from collections import defaultdict, Counter
from functools import partial
from tqdm.auto import trange
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import random
import jax


STRIDE = 0.5
N_STRIDES = 128


def process_tokens(stride=STRIDE, n_strides=N_STRIDES, seed=0, n_batches=1_000):
    tokens_processed = 0
    n_features = haver_sae.sae.d_hidden
    activ_cache = jnp.zeros((n_features, n_strides), dtype=jnp.int32)
    num_activations = jnp.zeros((n_features, n_strides), dtype=jnp.int32)
    # jsae = eqx.filter_jit(lambda s, x: s.encode(x)[1])
    cpu_device = jax.devices("cpu")[0]
    to_cpu = lambda x: np.asarray(jax.device_put(x, cpu_device))
    
    def get_active(feature, stride_idx, hiddens, mask):
        lower, upper = stride * stride_idx, stride * (stride_idx + 1)
        acts = hiddens[:, feature]
        # index = jnp.nonzero((acts >= lower) & (acts <= upper) & mask, size=1, fill_value=-1)[0]
        def scanner(carry, x):
            i, act, mask = x
            return jax.lax.cond((act >= lower) & (act <= upper) & (carry == -1) & mask,
                         lambda _: i,
                         lambda x: x,
                         operand=carry), None
        indices = jnp.arange(len(hiddens))
        # almost 2 OOMs slower
        # indices = jax.random.permutation(jax.random.key(feature), indices)
        indices = jax.random.permutation(jax.random.key(0), indices)
        return jax.lax.scan(scanner, -1, (indices, acts[indices], mask[indices]))[0]
    fun = jax.vmap(jax.vmap(get_active, in_axes=(None, 0, None, None), out_axes=0), in_axes=(0, None, None, None), out_axes=0)
    
    @partial(jax.jit, donate_argnums=(2, 3))
    def step(hiddens, mask, activ_cache, num_activations, tokens_processed):
        actives = fun(
            jnp.arange(n_features), jnp.arange(n_strides),
            hiddens,
            mask
        )
        probs = 1 / ((1 + num_activations).astype(jnp.float32) + 1e-12)
        probs = probs * (actives != -1)
        choose = jax.random.bernoulli(jax.random.PRNGKey(random.randint(0, 2**32 - 1)),
                                        probs)
        activ_cache = jnp.where(choose, actives + tokens_processed, activ_cache)
        num_activations += (actives != -1)
        return activ_cache, num_activations
    
    random.seed(seed)
    try:
        for _, texts in zip((bar := trange(n_batches)),
                            chunked(haver.create_dataset(), batch_size)):
            activations, model_misc = haver.model(texts)
            mask = model_misc.get("mask")
            hiddens = haver_sae.sae.encode(activations)[1]
            activ_cache, num_activations = step(
                hiddens, mask, activ_cache, num_activations,
                tokens_processed)

            tokens_processed += len(mask)
            bar.set_postfix(tokens_processed=tokens_processed, tps=tokens_processed / bar.format_dict["elapsed"])
    except KeyboardInterrupt:
        pass
    return to_cpu(activ_cache), to_cpu(num_activations)
activ_cache, num_activations = process_tokens(n_batches=n_batches)

# In[64]:

rng = 10
feature = 12
print(num_activations[feature])
for i, p in enumerate(activ_cache[feature]):
    if p == 0:
        continue
    tokens = token_table[max(0, p - rng):p+1]["tokens"].to_numpy()
    print((i * STRIDE, (i + 1) * STRIDE),
          repr(haver.model.decode(tokens)))


# In[67]:

from tqdm.auto import tqdm
import pyarrow as pa
import numpy as np

batches = []
for feat, activs in tqdm(activ_cache.items()):
    batches.append(pa.RecordBatch.from_pylist([dict(feature=feat, token=i, activation=np.float16(a))
                                               for i, a in activs], schema=pa.schema([
        ("feature", pa.int32()),
        ("token", pa.int32()),
        ("activation", pa.float16()),
    ])))

    # # 60% more efficient compression scheme, not guaranteed to work
    # token_0 = activs[0][0]
    # a_0 = activs[0][1]
    # schema = pa.schema([
    #     ("feature", pa.int32()),
    #     ("token", pa.uint16()),
    #     ("activation", pa.float16()),
    # ])
    # batches.append(pa.RecordBatch.from_pylist([dict(feature=feat, token=token_0, activation=np.float16(a_0))], schema=schema))
    # batches.append(pa.RecordBatch.from_pylist([dict(feature=feat, token=i2 - i1, activation=np.float16(a2))
    #                                            for (i1, a1), (i2, a2) in zip(activs[:-1], activs[1:])], schema=schema))


# In[68]:


import pyarrow.parquet as pq
import pyarrow as pa
pq_path = f"../weights/phi-l{layer}-activations.parquet"
table = pa.Table.from_batches(batches)
pq.write_table(table, pq_path, compression="snappy")


# In[ ]:


def visualize(feature, thresh=6.0):
    cache = activ_cache[feature]
    if not cache:    
        return
    tokens, activs = zip(*cache)
    if max(activs) < thresh:
        return
    freq = len(tokens) / tokens_processed
    print(freq)
    if freq > 0.03:
        return
    tokens_viewed = 0
    sli = 24
    for texts in chunked(tqdm(haver.create_dataset()), batch_size):
        toks = haver.model.to_tokens(texts)
        all_tokens = [t for tok in toks for t in tok]
        proc = sum(map(len, toks))
        all_token_ids = [tokens_viewed + i for i in range(proc)]
        for i, t in enumerate(all_token_ids):
            if t in tokens:
                activ = activs[tokens.index(t)]
                if activ < thresh:
                    continue
                print(activ, repr(haver.model.decode(all_tokens[max(0, i - sli + 1):i+1])),
                      repr(haver.model.decode(all_tokens[i+1:i+5])))
        tokens_viewed += proc
        if tokens_viewed > max(tokens):
            break
for i in range(10_000, 10_0100):
    visualize(i)
