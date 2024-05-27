#!/usr/bin/env python
# coding: utf-8

# In[1]:

try:
    get_ipython().run_line_magic('load_ext', 'autoreload')
    get_ipython().run_line_magic('autoreload', '2')
except NameError:
    pass

from saex.iterable_dataset import IterableDatasetConfig
from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.haver import ModelHaver, SAEHaver
from saex.sae import SAEConfig
from more_itertools import chunked

from micrlhf.utils.load_sae import get_sae
import os

from collections import defaultdict, Counter
from functools import partial
from tqdm.auto import trange
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import random
import jax

import pyarrow.parquet as pq
import pyarrow as pa

import pyarrow as pa
from tqdm.auto import trange
import pyarrow.parquet as pq

from tqdm.auto import tqdm
import pyarrow as pa
import numpy as np

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

@partial(jax.jit, donate_argnums=(2, 3), static_argnums=(5,))
def step(hiddens, mask, activ_cache, num_activations, tokens_processed, n_features):
    actives = fun(
        jnp.arange(n_features), jnp.arange(n_strides),
        hiddens,
        mask
    )
    probs = 1 / ((1 + num_activations).astype(jnp.float32) + 1e-20)
    probs = probs * (actives != -1)
    choose = jax.random.bernoulli(jax.random.PRNGKey(random.randint(0, 2**32 - 1)),
                                    probs)
    activ_cache = jnp.where(choose, actives + tokens_processed, activ_cache)
    num_activations += (actives != -1)
    return activ_cache, num_activations

# @jax.jit
def get_nonzeros(hiddens, max_l0):
    val, ind = jax.lax.top_k(jnp.abs(hiddens), max_l0)
    nonzeros = jnp.where(val != 0, ind, -1)
    return nonzeros

def main(layer=20, revision=8,
         n_batches=1000, tokens_file="tokens",
         mnt=64):
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
        activ_cache = defaultdict(list)
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

                indices = jnp.nonzero(mask)[0]
                hiddens = hiddens[indices]
                nonzeros = get_nonzeros(hiddens, max_l0=512)
                nonzeros = np.asarray(to_cpu(nonzeros.astype(jnp.int32)))
                hiddens = np.asarray(to_cpu(hiddens.astype(jnp.float16)))
                # mask = np.asarray(to_cpu(mask))
                indices = np.asarray(to_cpu(indices))
                # for i, h in zip(list(jnp.arange(len(hiddens))[mask]), list(hiddens[mask])):
                #     active_features = np.nonzero(h)[0]
                #     feature_activations = h[active_features]
                #     for f, a in zip(active_features, feature_activations):
                #         activ_cache[int(f)].append((tokens_processed + i, float(a)))


                for i, active_features, h in zip(indices, nonzeros, hiddens):
                    active_features = active_features[active_features != -1]
                    feature_activations = h[active_features]
                    for f, a in zip(active_features, feature_activations):
                        activ_cache[int(f)].append((i + tokens_processed, float(a)))
                tokens_processed += len(mask)
                assert i < len(mask)
                bar.set_postfix(tokens_processed=tokens_processed, tps=tokens_processed / bar.format_dict["elapsed"])

                tokens_processed += len(mask)
                bar.set_postfix(tokens_processed=tokens_processed, tps=tokens_processed / bar.format_dict["elapsed"])
        except KeyboardInterrupt:
            pass
        return activ_cache, tokens_processed
    activ_cache, tokens_processed = process_tokens(n_batches=n_batches)

    # In[67]:

    batches = []
    for feat, all_activs in tqdm(activ_cache.items()):
        use_list = set()
        freqs = dict()
        for st in range(n_strides):
            lower, upper = stride * st, stride * (st + 1)
            in_stride = [a for a in all_activs if lower <= a[1] < upper]
            if not len(in_stride):
                continue
            one_choice = random.choice(in_stride)[0]
            freq = len(in_stride) / tokens_processed
            for token in range(one_choice-mnt, one_choice+mnt):
                use_list.add(token)
                freqs[token] = freq
        for token, activ in all_activs:
            if token not in use_list:
                continue
            batches.append(pa.RecordBatch.from_pylist([dict(feature=feat,
                                                            token=token,
                                                            activation=np.float16(activ),
                                                            freq=np.float16(freqs[token]))],
                                                    schema=pa.schema([
                ("feature", pa.int32()),
                ("activation", pa.float16()),
                ("token", pa.int32()),
                ("freq", pa.float16()),
            ])))


    # In[68]:
    os.makedirs("weights/caches", exist_ok=True)
    pq_path = f"weights/caches/phi-l{layer}-r{revision}-st{stride}x{n_strides}-activations.parquet"
    table = pa.Table.from_batches(batches)
    pq.write_table(table, pq_path, compression="snappy")

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
for layer, revision in [(12, 8)]:
    haver.model.set_layer(layer)
    main(layer, revision, n_batches=20)
