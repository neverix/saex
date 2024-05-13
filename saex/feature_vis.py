from more_itertools import chunked

from .haver import ModelHaver
from .iterable_dataset import IterableDatasetConfig
from .models.micrlhf_model import MicrlhfModelConfig
from .sae import SAEConfig


def main():
    n_features = 3072
    batch_size = 16
    sae_config = SAEConfig(
        n_dimensions=n_features,
        sparsity_loss_type="l1",
        sparsity_coefficient=0,
        batch_size=batch_size,
        expansion_factor=32,
        use_encoder_bias=True,
        remove_decoder_bias=False,
        encoder_init_method="orthogonal",
        decoder_init_method="pseudoinverse",
        decoder_bias_init_method="zeros",
        reconstruction_loss_type="mse_batchnorm",
        project_updates_from_dec=True,
        death_loss_type="dm_ghost_grads",
        death_penalty_threshold=5e-7,
        death_penalty_coefficient=0.25,
        dead_after=1_000,
        buffer_size=2_000,
        restrict_dec_norm="exact",
        sparsity_tracking_epsilon=0.1,
        is_gated=True,
    )
    dataset_config = IterableDatasetConfig(
        dataset_name="nev/openhermes-2.5-phi-format-text",
    )
    model_config = MicrlhfModelConfig(
        tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
        gguf_path="weights/phi-3-16.gguf",
        device_map="tpu:0",

        layer=11,
        max_seq_len=128,
    )
    haver = ModelHaver(model_config=model_config, sae_config=sae_config,
                       dataset_config=dataset_config,
                       sae_restore="weights/phi-l11-gated.safetensors")
    for texts in chunked(haver.create_dataset()):
        activations, model_misc = haver.model(texts)
        mask = model_misc.get("mask")
        _, hiddens = haver.sae.encode(activations)
        print(hiddens.shape)
        break


if __name__ == "__main__":
    main()
