import os
from dataclasses import dataclass, field
from typing import List

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
    device_map: str = "auto"

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
        self._llama_residuals_call = jax.jit(lambda lr, inputs: lr(inputs)[1][0].value)
        self.sharding = jshard.NamedSharding(self.mesh, jshard.PartitionSpec("dp", None))

    def __call__(self, texts: List[str]):
        tokens = self.encode_texts(texts)
        token_array = pz.nx.wrap(jnp.asarray(tokens["input_ids"]).reshape((1, -1)), "batch", "seq").untag("batch").repeat(512).tag("batch")
        inputs = self._llama.inputs.from_basic_segments(token_array)
        hidden_states = self._llama_residuals_call(self._llama_residuals, inputs)
        hidden_states = hidden_states.untag("batch", "seq", "embedding").data_array

        mask = tokens["attention_mask"].reshape(-1)
        return hidden_states.reshape(-1, hidden_states.shape[-1]), {"mask": mask}

    def encode_texts(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="jax", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        tokens = {k: jax.device_put(v, self.sharding) for k, v in tokens.items()}
        return tokens


if __name__ == "__main__":
    MicrlhfModelConfig(
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
        gguf_path="weights/phi-3-16.gguf"
    )
