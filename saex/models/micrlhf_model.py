import os
from dataclasses import dataclass, field, replace
from typing import List

import jax
import jax.numpy as jnp
import jax.sharding as jshard
import jax.tree_util

import equinox as eqx
from micrlhf.llama import LlamaBlock, LlamaTransformer
from penzai import pz
from penzai.toolshed import jit_wrapper
from transformers import AutoTokenizer


@pz.pytree_dataclass
class ScanSequential(pz.Layer):
    layer: pz.Layer

    # @jax.jit
    def __call__(self, inputs):
        return jax.lax.scan(
            lambda h, l: (l(h), None),
            inputs,
            jax.tree_map(lambda x: x.untag("layer"), self.layer))[0]


def sequential_to_scan(model):
    def fn(seq):
        layers = seq.sublayers
        layers = [l for l in layers if not isinstance(l, pz.nn.Identity)]
        layer = jax.tree_map(lambda *xs: pz.nx.stack(xs, "layer"), *layers)
        return ScanSequential(layer)

    return model.select().at_instances_of(pz.nn.Sequential).pick_nth_selected(1).apply(fn)


@dataclass
class MicrlhfModelConfig:
    tokenizer_path: os.PathLike
    gguf_path: os.PathLike
    layer: int
    max_seq_len: int = 512
    device_map: str = "auto:mp=2"

    @property
    def model_class(self) -> type:
        return MicrlhfModel


class MicrlhfModel(object):
    has_mesh: bool = True
    
    def __init__(self, config: MicrlhfModelConfig):
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self._llama = LlamaTransformer.from_pretrained(config.gguf_path, device_map=config.device_map)
        self.mesh = self._llama.mesh
        tag = f"residual-{config.layer}"
        self._llama_residuals = pz.de.CollectingSideOutputs.handling(sequential_to_scan(
            self._llama.select()
            .at_instances_of(LlamaBlock)
            .pick_nth_selected(config.layer)
            .insert_before(pz.de.TellIntermediate.from_config(tag=tag)),
        ))
        self._llama_residuals_call = jax.jit(lambda lr, inputs: lr(inputs))
        self._llama_replace = sequential_to_scan(
            self._llama.select()
            .at_instances_of(LlamaBlock)
            .apply_with_selected_index(
                lambda i, x: x if i >= config.layer else pz.nn.Identity()
            )
            .select()
            .at_instances_of(pz.nn.EmbeddingLookup)
            .apply(lambda _: pz.nn.Identity())
        )
        self._llama_replace_call = jax.jit(
            (lambda sae_s, sae_d, inputs, lr, hiddens:
                lr(replace(inputs, tokens=
                           pz.nx.wrap(eqx.combine(sae_s, sae_d).forward(hiddens), "batch", "seq", "embedding")))),
            static_argnums=0)
        @jax.jit
        def loss_fn(logits, inputs, masks):
            logits = pz.nx.nmap(lambda l, i, m: jnp.take_along_axis(jax.nn.log_softmax(l[:-1], -1), i[1:, None], 1)[:, 0] * m[1:]
                                )(logits.untag("seq", "vocabulary"), inputs.tokens.untag("seq"), masks.untag("seq"))
            return -logits.data_array.mean() / masks.data_array.mean()
        self.loss_fn = loss_fn
        self.sharding = jshard.NamedSharding(self.mesh, jshard.PartitionSpec("dp", None))

    def __call__(self, texts: List[str]):
        inputs, mask = self.encode_texts(texts)
        hidden_states = self._llama_residuals_call(self._llama_residuals, inputs)[1][0].value
        hidden_states = hidden_states.untag("batch", "seq", "embedding").data_array

        return hidden_states.reshape(-1, hidden_states.shape[-1]), {"mask": mask}

    def eval_loss(self, texts, autoencoder):
        inputs, mask = self.encode_texts(texts)
        logits, (hidden_states,) = self._llama_residuals_call(self._llama_residuals, inputs)
        hidden_states = hidden_states.value.untag("batch", "seq", "embedding").data_array
        sae_params, sae_static = eqx.partition(autoencoder, eqx.is_array)
        logits_reconstructed = self._llama_replace_call(
            sae_static, sae_params, inputs, self._llama_replace, hidden_states)

        mask = pz.nx.wrap(mask.reshape(inputs.tokens.data_array.shape), *inputs.tokens.named_axes.keys())
        loss = self.loss_fn(logits, inputs, mask)
        loss_reconstructed = self.loss_fn(logits_reconstructed, inputs, mask)

        return loss, loss_reconstructed

    def to_str_tokens(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="jax", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        return [self._tokenizer.batch_decode(ii) for ii in tokens["input_ids"]]

    def to_tokens(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="jax", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        return tokens["input_ids"]

    def decode(self, tokens: List[int]):
        return self._tokenizer.decode(list(map(int, tokens)))

    def encode_texts(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="jax", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        tokens = {k: jax.device_put(v, self.sharding) for k, v in tokens.items()}
        token_array = pz.nx.wrap(jnp.asarray(tokens["input_ids"]).reshape((1, -1)), "batch", "seq")
        inputs = self._llama.inputs.from_basic_segments(token_array)
        return inputs, tokens["attention_mask"].reshape(-1)


if __name__ == "__main__":
    MicrlhfModelConfig(
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
        gguf_path="weights/phi-3-16.gguf"
    )
