import os
from dataclasses import dataclass, field, replace
from typing import List

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
from micrlhf.llama import LlamaBlock, LlamaTransformer
from penzai import pz
from penzai.toolshed import jit_wrapper
from transformers import AutoTokenizer


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
        self._llama_residuals = pz.de.CollectingSideOutputs.handling(
            self._llama.select()
            .at_instances_of(LlamaBlock)
            .pick_nth_selected(config.layer)
            .insert_before(pz.de.TellIntermediate.from_config(tag=tag)),
        )
        self._llama_residuals_call = jax.jit(lambda lr, inputs: lr(inputs))
        self._llama_replace = pz.de.WithSideInputsFromInputTuple.handling(
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
                lr(replace(inputs, tokens=eqx.combine(sae_s, sae_d).forward(hiddens)))),
            static_argnums=0)
        self.sharding = jshard.NamedSharding(self.mesh, jshard.PartitionSpec("dp", None))

    def __call__(self, texts: List[str]):
        inputs, mask = self.encode_texts(texts)
        hidden_states = self._llama_residuals_call(self._llama_residuals, inputs)
        hidden_states = hidden_states.untag("batch", "seq", "embedding").data_array

        return hidden_states.reshape(-1, hidden_states.shape[-1]), {"mask": mask}

    def eval_loss(self, texts, autoencoder):
        inputs, mask = self.encode_texts(texts)
        logits, (hidden_states,) = self._llama_residuals_call(self._llama_residuals, inputs)
        hidden_states = hidden_states.value
        assert (self.config.cache_n == 0) or (self._cache[0] is not None)
        sae_params, sae_static = eqx.partition(autoencoder, eqx.is_array)
        logits_reconstructed = self._llama_replace_call(
            sae_static, sae_params, self._llama_replace, inputs, self._llama_replace, hidden_states)

        loss = self.loss_fn(logits, inputs)
        loss_reconstructed = self.loss_fn(logits_reconstructed, inputs)

        return loss, loss_reconstructed

    def loss_fn(logits, inputs):
        logits = pz.nx.nmap(lambda l, i: jnp.take_along_axis(jax.nn.log_softmax(l[:-1]), 1, i[1:]))(logits.untag("seq", "vocab"), inputs.tokens.untag("seq"))
        return logits.data_array.mean()

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
