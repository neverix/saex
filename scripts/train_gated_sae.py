import fire
import jax
import jax.numpy as jnp

from saex.train_script import train_main
from saex.trainer_cache import (BufferTrainerConfig, IterableDatasetConfig,
                                SAEConfig, TransformersModelConfig)


def main(
    n_devices: int = len(jax.devices()),
    mp_devices: int = 1,
    cache_size = 2**19,
    cache_batch_size = 128,
    batch_size = 1024,
    max_seq_len = 1024,
    sparsity_coefficient=8e-5,
    save_steps=1000,
    layer = 1,
    is_xl=False,
    restore = False,
    train_on_lamini = False,
    wandb_entity = "neverix",
):
    n_features = 768 if not is_xl else 1600

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
        save_path=f"weights/gpt2-{layer}-gated.safetensors" if not restore else f"weights/gpt2-{layer}-gated-tuned.safetensors",
        dry_run_steps=0,
        no_update=False,
        sae_config=SAEConfig(
            n_dimensions=n_features,
            # sparsity_loss_type="l1",
            # sparsity_loss_type="l1_sqrt",
            sparsity_loss_type="recip",
            recip_schedule = ((100_000, 0.1),),
            sparsity_coefficient=sparsity_coefficient,
            batch_size=batch_size,
            expansion_factor=32,
            use_encoder_bias=True,
            remove_decoder_bias=False,
            encoder_init_method="orthogonal",
            decoder_init_method="pseudoinverse",
            decoder_bias_init_method="zeros",
            # decoder_bias_init_method="geom_median" if not restore else "zeros",
            reconstruction_loss_type="mse_batchnorm",
            project_updates_from_dec=True,
            death_loss_type="dm_ghost_grads",
            # death_loss_type="sparsity_threshold",
            # death_loss_type="none",
            death_penalty_threshold=1e-6,
            death_penalty_coefficient=0.25,
            dead_after=1000,
            restrict_dec_norm="exact",
            sparsity_tracking_epsilon=0.1,
            # is_gated=True,
            is_gated=False,
        ),
        sae_restore=restore,
        cache_every_steps=int(cache_size / batch_size),
        cache_batch_size=cache_batch_size,
        cache_acc=int(cache_size / cache_batch_size / max_seq_len),
        buffer_max_samples=cache_size,
        restore_buffer=False,
        save_buffer=False,
        model_config=TransformersModelConfig(
            model_name_or_path="gpt2" if not is_xl else "openai-community/gpt2-xl",
            # model_name_or_path="MBZUAI/LaMini-GPT-124M",
            from_pt=True,
            layer=layer,
            max_seq_len=max_seq_len,
            add_prefix="<|endoftext|>",
            concat_all=False,
            
            cache_n=25 if train_on_lamini else 0,
            return_real_mask=True,
        ),
        dataset_config=IterableDatasetConfig(
            dataset_name="Skylion007/openwebtext" if not train_on_lamini else "nev/lamini-dataset-text",
        ),
        loss_batch_size=16,
        eval_loss_every=900_000,  # loss evaluation takes too long for GPT-2 (flax)
        buffer_dtype=jnp.float16,
        use_devices=n_devices,
        mp_devices=mp_devices,
    )
    train_main(config)


if __name__ == "__main__":
    fire.Fire(main)
