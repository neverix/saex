import fire
import jax
import jax.numpy as jnp

from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.train_script import train_main
from saex.trainer_cache import (BufferTrainerConfig, IterableDatasetConfig,
                                SAEConfig)


def main(
    n_devices: int = 1,
    mp_devices: int = 1,
    cache_size = 2**14,
    cache_batch_size = 32,
    batch_size = 1024,
    max_seq_len = 256,
    sparsity_coefficient=5e-6,
    save_steps=0,
    restore = False,
    wandb_entity = "neverix",
    layer = 10,
):
    n_features = 3072

    config = BufferTrainerConfig(
        n_dimensions=n_features,
        lr=4e-4,  # Higher LR limits the variance explained but can lead to faster decrease in L0
        beta1=0.0,  # Crucial for avoiding dead features
        beta2=0.99,  # Lower beta2 can lead to a slightly faster decrease in L0
        scheduler_warmup=128,
        scheduler_cycle=100_000,
        scheduler_multiply=0.1,
        train_iterations=100_000,
        save_steps=save_steps,
        use_wandb=(wandb_entity, "saex") if wandb_entity else None,
        log_every=10,
        hist_every=100,
        save_path=f"weights/phi-l{layer}.safetensors",
        dry_run_steps=0,
        no_update=False,
        sae_config=SAEConfig(
            n_dimensions=n_features,
            sparsity_loss_type="recip",
            recip_schedule = ((100_000, 0.1),),
            sparsity_coefficient=sparsity_coefficient,
            batch_size=batch_size,
            expansion_factor=8,
            use_encoder_bias=True,
            remove_decoder_bias=restore,
            encoder_init_method="orthogonal",
            decoder_init_method="pseudoinverse",
            decoder_bias_init_method="zeros",
            # decoder_bias_init_method="geom_median" if not restore else "zeros",
            reconstruction_loss_type="mse_batchnorm",
            project_updates_from_dec=True,
            death_loss_type="dm_ghost_grads",
            death_penalty_threshold=1e-5,
            death_penalty_coefficient=1,
            dead_after=500,
            restrict_dec_norm="exact",
            sparsity_tracking_epsilon=0.05,
        ),
        sae_restore=restore,
        cache_every_steps=int(cache_size / batch_size),
        cache_batch_size=cache_batch_size,
        cache_acc=int(cache_size / cache_batch_size / max_seq_len),
        buffer_max_samples=cache_size,
        restore_buffer=False,
        save_buffer=False,
        model_config=MicrlhfModelConfig(
            tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
            gguf_path="weights/phi-3-16.gguf",
            device_map="auto" if n_devices > 1 else "tpu:0",

            layer=layer,
            max_seq_len=max_seq_len,
        ),
        dataset_config=IterableDatasetConfig(
            dataset_name="nev/openhermes-2.5-phi-format-text",
        ),
        loss_batch_size=16,
        eval_loss_every=900_000,
        buffer_dtype=jnp.bfloat16,
        use_devices=n_devices,
        mp_devices=mp_devices,
    )
    train_main(config)


if __name__ == "__main__":
    fire.Fire(main)
