#!/usr/bin/env python
# coding: utf-8

# In[1]:

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

import dataclasses
import os
import random
from collections import Counter, defaultdict
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from micrlhf.utils.load_sae import get_sae
from more_itertools import chunked
from tqdm.auto import tqdm, trange

from saex.haver import ModelHaver, SAEHaver
from saex.iterable_dataset import IterableDatasetConfig
from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.sae import SAEConfig

stride = 0.25
n_strides = 128

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

@partial(jax.jit, donate_argnums=(2, 3, 6,), static_argnums=(5, 7,))
def step(hiddens, mask, activ_cache, num_activations, tokens_processed, n_features, nearby_activations, nearby):
    actives = fun(
        jnp.arange(n_features), jnp.arange(n_strides),
        hiddens,
        mask
    )
    probs = 1 / ((1 + num_activations).astype(jnp.float32) + 1e-20)
    probs = probs * (actives != -1)
    choose = jax.random.bernoulli(jax.random.PRNGKey(random.randint(0, 2**32 - 1)),
                                    probs)
    nearby_activs = jnp.take_along_axis(
        hiddens[:, :, None, None],
        actives[None, :, :, None] + jnp.arange(-nearby, nearby + 1),
        axis=0)[0]
    nearby_activs = (nearby_activs / stride).astype(jnp.uint8)
    activ_cache = jnp.where(choose, actives + tokens_processed, activ_cache)
    nearby_activations = jnp.where(choose[..., None], nearby_activs, nearby_activations)
    num_activations += (actives != -1)
    return activ_cache, num_activations, nearby_activations

def main(layer=20, revision=8,
         n_batches = 1000, tokens_file="tokens",
         nearby=8):
    # In[2]:

    os.makedirs("weights/sae", exist_ok=True)
    sae_path = get_sae(layer, revision, model_dir="weights", return_fname=True)
    haver_sae = SAEHaver(
        sae_config=sae_config,
        mesh=haver.mesh,
        sae_restore=sae_path)

    # In[64]:

    token_batches = []
    for _, texts in zip(trange(n_batches), chunked(haver.create_dataset(), batch_size)):
        tokens = haver.model.to_tokens(texts)
        token_batches.append(pa.RecordBatch.from_arrays([tokens.reshape(-1)], ["tokens"]))
    token_table = pa.Table.from_batches(token_batches)
    pq.write_table(token_table, f"weights/{tokens_file}.parquet",
                compression="snappy")

    # In[63]:

    def process_tokens(seed=0, n_batches=1_000):
        tokens_processed = 0
        n_features = haver_sae.sae.d_hidden
        activ_cache = jnp.zeros((n_features, n_strides), dtype=jnp.int32)
        num_activations = jnp.zeros((n_features, n_strides), dtype=jnp.int32)
        nearby_activations = jnp.zeros((n_features, n_strides, nearby * 2 + 1), dtype=jnp.uint8)
        # jsae = eqx.filter_jit(lambda s, x: s.encode(x)[1])
        cpu_device = jax.devices("cpu")[0]
        to_cpu = lambda x: np.asarray(jax.device_put(x, cpu_device))
        
        random.seed(seed)
        try:
            for _, texts in zip((bar := trange(n_batches)),
                                chunked(haver.create_dataset(), batch_size)):
                activations, model_misc = haver.model(texts)
                mask = model_misc.get("mask")
                hiddens = haver_sae.sae.encode(activations)[1]
                activ_cache, num_activations, nearby_activations = step(
                    hiddens, mask, activ_cache, num_activations,
                    tokens_processed, n_features,
                    nearby_activations, nearby)

                tokens_processed += len(mask)
                bar.set_postfix(tokens_processed=tokens_processed, tps=tokens_processed / bar.format_dict["elapsed"])
        except KeyboardInterrupt:
            pass
        return to_cpu(activ_cache), to_cpu(num_activations), to_cpu(nearby_activations), tokens_processed
    activ_cache, num_activations, nearby_activations, tokens_processed = process_tokens(n_batches=n_batches)

    # # In[64]:

    # rng = 10
    # feature = 12
    # print(num_activations[feature])
    # for i, p in enumerate(activ_cache[feature]):
    #     if p == 0:
    #         continue
    #     tokens = token_table[max(0, p - rng):p+1]["tokens"].to_numpy()
    #     print((i * stride, (i + 1) * stride),
    #         repr(haver.model.decode(tokens)))

    # # In[67]:

    batches = []
    for feat, (activs, freqs, nearbies) in enumerate(zip(tqdm(activ_cache),
                                                         num_activations,
                                                         nearby_activations)):
        # total_activs = freqs.sum()
        batches.append(pa.RecordBatch.from_pylist([dict(feature=feat, token=i,
                                                        activation=np.float16(j * stride),
                                                        freq=np.float16(f / (tokens_processed / batch_size)),
                                                        nearby=n.tolist())
                                                for j, (i, f, n) in enumerate(zip(activs, freqs, nearbies))],
                                                schema=pa.schema([
            ("feature", pa.int32()),
            ("activation", pa.float16()),
            ("token", pa.int32()),
            ("freq", pa.float16()),
            ("nearby", pa.list_(pa.uint8()))
        ])))


    # In[68]:
    os.makedirs("weights/caches", exist_ok=True)
    pq_path = f"weights/caches/phi-l{layer}-r{revision}-st{stride}x{n_strides}-activations.parquet"
    table = pa.Table.from_batches(batches)
    pq.write_table(table, pq_path, compression="snappy")
    os.remove(sae_path)


n_features = 3072
batch_size = 64
max_seq_len = 128
dataset_config = IterableDatasetConfig(
    dataset_name="nev/openhermes-2.5-phi-format-text",
    # dataset_name="nev/generated-phi-format-text",
)
model_config = MicrlhfModelConfig(
    tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
    gguf_path="weights/phi-3-16.gguf",
    device_map="auto",
    use_flash=False,
    layer=0,
    max_seq_len=max_seq_len,
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
for layer, revision in [(10, 4)]:
    haver.model.set_layer(layer)
    sae_config = dataclasses.replace(sae_config, is_gated=revision < 6)
    sae_config = dataclasses.replace(sae_config, expansion_factor=32 if revision >= 6 else 16)
    main(layer, revision, n_batches=1000)
