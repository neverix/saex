import os
from dataclasses import dataclass, field, replace
from typing import List, Literal

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.sharding as jshard
import jax.tree_util
from micrlhf.flash import flashify
from micrlhf.llama import LlamaBlock, LlamaTransformer, LlamaMLP
from micrlhf.scan import sequential_to_scan
from penzai import pz
from penzai.toolshed import jit_wrapper
from transformers import AutoTokenizer


@dataclass
class MicrlhfModelConfig:
    tokenizer_path: os.PathLike
    gguf_path: os.PathLike
    layer: int
    sae_type: Literal["residual", "attn_out", "transcoder"] = "residual"
    max_seq_len: int = 512
    device_map: str = "auto:mp=2"
    use_flash: bool = False
    from_type: Literal[None, "gemma"] = None
    load_eager: bool = True

    @property
    def has_outputs(self):
        return self.sae_type == "transcoder"

    @property
    def model_class(self) -> type:
        return MicrlhfModel


@pz.pytree_dataclass
class SideInput(pz.Layer):
    value: pz.de.SideInputRequest[pz.nx.NamedArrayBase]

    @classmethod
    def from_config(cls, tag: str):
        return cls(pz.de.SideInputRequest(tag))

    def __call__(self, inputs):
        return self.value.ask()


def remove_head(model):
    model = model.select() \
        .at_instances_of(pz.nn.EmbeddingDecode) \
        .apply(lambda _: pz.nn.Identity())
    model = model.select().at_instances_of(pz.nn.RMSLayerNorm)
    model = model.pick_nth_selected(model.count() - 1).apply(lambda _: pz.nn.Identity())
    return model


class MicrlhfModel(object):
    has_mesh: bool = True
    
    def __init__(self, config: MicrlhfModelConfig):
        self.config = config
        self._tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path)
        self._llama = LlamaTransformer.from_pretrained(config.gguf_path, device_map=config.device_map,
                                                       from_type=config.from_type, load_eager=config.load_eager)
        if config.use_flash:
            self._llama = flashify(self._llama)
        self.mesh = self._llama.mesh
        self.set_layer(config.layer)
        @jax.jit
        def loss_fn(logits, inputs, masks):
            logits = pz.nx.nmap(lambda l, i, m: jnp.take_along_axis(jax.nn.log_softmax(l[:-1], -1), i[1:, None], 1)[:, 0] * m[1:]
                                )(logits.untag("seq", "vocabulary"), inputs.tokens.untag("seq"), masks.untag("seq"))
            return -logits.data_array.mean() / masks.data_array.mean()
        self.loss_fn = loss_fn
        self.sharding = jshard.NamedSharding(self.mesh, jshard.PartitionSpec("dp", None))

    def set_layer(self, layer):
        if self.config.sae_type == "residual":
            tag = f"residual-{layer}"
            get_residuals = self._llama.select() \
                .at_instances_of(LlamaBlock) \
                .pick_nth_selected(layer) \
                .insert_before(pz.de.TellIntermediate.from_config(tag=tag))
            self._getter = pz.de.CollectingSideOutputs.handling(get_residuals)
            get_residuals_fast = self._llama.select() \
                .at_instances_of(LlamaBlock) \
                .apply_with_selected_index(
                    lambda i, x: x if i < layer else pz.nn.Identity()
                )
            self._value = remove_head(get_residuals_fast)
            self._value_call = jax.jit(lambda lr, inputs: lr(inputs))
            
            #self._value_call = jax.jit(lambda lr, inputs: lr(inputs)[1][0].value)
            #self._value = self._getter
            
            self._getter_call = jax.jit(lambda lr, inputs: lr(inputs))
            replaced = \
                self._llama.select() \
                .at_instances_of(LlamaBlock) \
                .apply_with_selected_index(
                    lambda i, x: x if i >= layer else pz.nn.Identity()
                ) \
                .select() \
                .at_instances_of(pz.nn.EmbeddingLookup) \
                .apply(lambda _: pz.nn.Identity())
            if self.config.from_type == "gemma":
                replaced = replaced.select().at_instances_of(pz.nn.ConstantRescale
                                                            ).pick_nth_selected(0).apply(lambda _: pz.nn.Identity())
            self._setter = sequential_to_scan(replaced)
            self._setter_call = jax.jit(
                (lambda sae_s, sae_d, inputs, lr, hiddens:
                    lr(replace(inputs, tokens=
                            pz.nx.wrap(eqx.combine(sae_s, sae_d).forward(hiddens), "batch", "seq", "embedding")))),
                static_argnums=0)
        elif self.config.sae_type == "attn_out":
            attention_cls = pz.nn.Attention
            tag = f"attn_out-{layer}"
            get_attns = self._llama.select() \
                .at_instances_of(pz.nn.Residual) \
                .apply_with_selected_index(
                    lambda i, x: (x.delta if i == layer * 2 else x) if i <= layer * 2 else pz.nn.Identity()
                )
            get_attns = remove_head(get_attns)
            get_residuals = self._llama.select() \
                .at_instances_of(attention_cls) \
                .pick_nth_selected(layer) \
                .apply(lambda x: pz.nn.Sequential([x, pz.de.TellIntermediate.from_config(tag=tag)]))
            self._getter = pz.de.CollectingSideOutputs.handling(get_residuals)
            self._value_call = jax.jit(lambda la, inputs: la(inputs))
            self._value = get_attns
            
            #self._value_call = jax.jit(lambda la, inputs: la(inputs)[1][0].value)
            #self._value = self._getter
            
            self._getter_call = jax.jit(lambda la, inputs: la(inputs))
            replaced = \
                self._llama.select() \
                .at_instances_of(attention_cls) \
                .pick_nth_selected(layer) \
                .apply(
                    lambda _: SideInput.from_config(tag)
                )
            self._setter = pz.de.WithSideInputsFromInputTuple.handling(replaced, [tag])
            self._setter_call = jax.jit(
                (lambda sae_s, sae_d, inputs, la, hiddens:
                    la((inputs, pz.nx.wrap(eqx.combine(sae_s, sae_d).forward(hiddens), "batch", "seq", "embedding"),))),
                static_argnums=0)
        elif self.config.sae_type == "transcoder":
            mlp_cls = LlamaMLP
            get_residuals = self._llama.select() \
                .at_instances_of(mlp_cls) \
                .pick_nth_selected(layer) \
                .apply(lambda x: pz.nn.Sequential([pz.de.TellIntermediate.from_config(tag=f"mlp-in-{layer}"), x, pz.de.TellIntermediate.from_config(tag=f"mlp-out-{layer}")]))
            self._getter = pz.de.CollectingSideOutputs.handling(get_residuals)
            self._value_call = jax.jit(lambda la, inputs: tuple(m.value for m in la(inputs)[1]))
            self._getter_call = jax.jit(lambda la, inputs: la(inputs))
            self._value = self._getter
            
            replaced = \
                self._llama.select() \
                .at_instances_of(mlp_cls) \
                .pick_nth_selected(layer) \
                .apply(
                    lambda _: SideInput.from_config(f"mlp-rep-{layer}")
                )
            self._setter = pz.de.WithSideInputsFromInputTuple.handling(replaced, [f"mlp-rep-{layer}"])
            self._setter_call = jax.jit(
                (lambda sae_s, sae_d, inputs, la, hiddens:
                    la((inputs, pz.nx.wrap(eqx.combine(sae_s, sae_d).forward(hiddens), "batch", "seq", "embedding"),))),
                static_argnums=0)
        else:
            raise ValueError(f"Invalid SAE type: {self.config.sae_type}")

    def __call__(self, texts: List[str]):
        inputs, mask = self.encode_texts(texts)
        hidden_states = self._value_call(self._value, inputs)
        if isinstance(hidden_states, tuple):
            hidden_states = jnp.concatenate(tuple(hs.untag("batch", "seq", "embedding").data_array for hs in hidden_states), axis=-1)
        else:
            hidden_states = hidden_states.untag("batch", "seq", "embedding").data_array

        return hidden_states.reshape(-1, hidden_states.shape[-1]), {"mask": mask}

    def eval_loss(self, texts, autoencoder):
        inputs, mask = self.encode_texts(texts)
        logits, hidden_states = self._getter_call(self._getter, inputs)
        hidden_states = hidden_states[0].value.untag("batch", "seq", "embedding").data_array
        sae_params, sae_static = eqx.partition(autoencoder, eqx.is_array)
        logits_reconstructed = self._setter_call(
            sae_static, sae_params, inputs, self._setter, hidden_states)

        mask = pz.nx.wrap(mask.reshape(inputs.tokens.data_array.shape), *inputs.tokens.named_axes.keys())
        loss = self.loss_fn(logits, inputs, mask)
        loss_reconstructed = self.loss_fn(logits_reconstructed, inputs, mask)

        return loss, loss_reconstructed

    def to_str_tokens(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="np", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        return [self._tokenizer.batch_decode(ii) for ii in tokens["input_ids"]]

    def to_tokens(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="np", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        return tokens["input_ids"]

    def decode(self, tokens: List[int]):
        return self._tokenizer.decode(list(map(int, tokens)))

    def encode_texts(self, texts: List[str]):
        tokens = self._tokenizer.batch_encode_plus(
            texts,
            return_tensors="np", padding="max_length",
            truncation=True, max_length=self.config.max_seq_len)
        tokens = {k: jax.device_put(v, self.sharding) for k, v in tokens.items()}
        token_array = pz.nx.wrap(jnp.asarray(tokens["input_ids"]), "batch", "seq")
        inputs = self._llama.inputs.from_basic_segments(token_array)
        return inputs, tokens["attention_mask"].reshape(-1)


if __name__ == "__main__":
    MicrlhfModelConfig(
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
        gguf_path="weights/phi-3-16.gguf"
    )
