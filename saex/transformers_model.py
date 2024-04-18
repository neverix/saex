import os
from dataclasses import dataclass

import transformers


class TransformersModel(object):
    # Extracts features from the residual stream

    def __init__(self, config: "TransformersModelConfig"):
        self.config = config
        self._model = transformers.FlaxAutoModel.from_pretrained(config.model_name_or_path)
        self._tokenizer = transformers.AutoTokenizer.from_pretrained(config.model_name_or_path)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._cache = (None, None)

    def __call__(self, texts):
        tokens = self._tokenizer(texts, return_tensors="jax", padding="max_length", truncation=True,
                                 max_length=self.config.max_seq_len)
        outputs = self._model(**tokens, output_hidden_states=True, past_key_values=self._cache[0])
        hidden_states = outputs.hidden_states[self.config.layer]
        if self.config.cache_n > 0 and self._cache[0] is None:
            self._cache = outputs.past_key_values, hidden_states
        # TODO but I don't think we should output cached hidden states; those never change

        return hidden_states.reshape(-1, hidden_states.shape[-1]), {"mask": tokens["attention_mask"].reshape(-1)}

    # def eval_loss(self, texts, autoencoder):
    #     tokens = self._tokenizer(texts, return_tensors="jax", padding="max_length", truncation=True,
    #                              max_length=self._model.config.n_positions)
    #     outputs = self._model(**tokens, output_hidden_states=True, past_key_values=self._cache[0])


@dataclass
class TransformersModelConfig:
    model_name_or_path: os.PathLike
    layer: int
    cache_n: int = 0
    cache_hidden_states: bool = False
    max_seq_len: int = 512
    @property
    def model_class(self) -> type:
        return TransformersModel
