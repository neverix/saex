import fire
import jax
import jax.numpy as jnp

from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.train_script import train_main
from saex.trainer_cache import (BufferTrainerConfig, IterableDatasetConfig,
                                SAEConfig)


def main(
    train_steps: int = 100_000,
    n_devices: int = 4,
    mp_devices: int = 1,
    cache_size = 2**16,
    cache_batch_size = 256,
    cache_ratio=2.0,
    batch_size = 2048,
    max_seq_len = 128,
    sparsity_coefficients=[4e-6],
    save_steps=2500,
    eval_loss_every=100,
    restore = False,
    wandb_entity = "neverix",
    layer = 20,
    is_gated: bool = True,
    use_recip=False,
    death_penalty_threshold=9e-5,
    push_to_hub=None,
    ema=None,
):
    n_features = 3072

    configs = []
    for sparsity_coefficient in sparsity_coefficients:
        config = BufferTrainerConfig(
            n_dimensions=n_features,
            lr=4e-4,  # Higher LR limits the variance explained but can lead to faster decrease in L0
            beta1=0.0,  # Crucial for avoiding dead features
            beta2=0.99,  # Lower beta2 can lead to a slightly faster decrease in L0
            scheduler_warmup=128,
            scheduler_cycle=100_000,
            scheduler_multiply=0.1,
            train_iterations=train_steps,
            save_steps=save_steps,
            use_wandb=(wandb_entity, "saex") if wandb_entity else None,
            log_every=10,
            hist_every=100,
            save_path=f"weights/phi-l{layer}{'-gated' if is_gated else ''}.safetensors",
            dry_run_steps=0,
            no_update=False,
            sae_config=SAEConfig(
                n_dimensions=n_features,
                sparsity_loss_type="recip" if use_recip else "l1",
                recip_schedule = ((100_000, 0.1),),
                sparsity_coefficient=sparsity_coefficient,
                batch_size=batch_size,
                expansion_factor=16,
                use_encoder_bias=True,
                remove_decoder_bias=False,
                encoder_init_method="orthogonal",
                decoder_init_method="pseudoinverse",
                decoder_bias_init_method="zeros",
                # decoder_bias_init_method="geom_median" if not restore else "zeros",
                reconstruction_loss_type="mse_batchnorm",
                project_updates_from_dec=True,
                death_loss_type="dm_ghost_grads",
                death_penalty_threshold=death_penalty_threshold,
                death_penalty_coefficient=0.25,
                dead_after=1_000,
                buffer_size=2_000,
                restrict_dec_norm="exact",
                sparsity_tracking_epsilon=0.1,
                is_gated=is_gated,
            ),
            sae_restore=restore,
            cache_every_steps=int(cache_size / batch_size * cache_ratio),
            cache_batch_size=cache_batch_size,
            cache_acc=int(cache_size / cache_batch_size / max_seq_len),
            buffer_max_samples=cache_size,
            restore_buffer=False,
            save_buffer=False,
            model_config=MicrlhfModelConfig(
                tokenizer_path="microsoft/Phi-3-mini-4k-instruct",
                gguf_path="weights/phi-3-16.gguf",
                device_map=f"auto:mp={mp_devices}" if n_devices > 1 else "tpu:0",
                use_flash=max_seq_len >= 128 and max_seq_len % 128 == 0,
                layer=layer,
                max_seq_len=max_seq_len,
            ),
            dataset_config=IterableDatasetConfig(
                dataset_name="nev/openhermes-2.5-lamini-phi-format-text",
            ),
            loss_batch_size=16,
            eval_loss_every=eval_loss_every,
            buffer_dtype=jnp.bfloat16,
            use_devices=n_devices,
            mp_devices=mp_devices,
            push_to_hub=None if push_to_hub is None else (push_to_hub[0], push_to_hub[1]
                                                          + f"-{sparsity_coefficient:.2E}"),
            ema=ema,
        )
        configs.append(config)
    train_main(configs)


if __name__ == "__main__":
    fire.Fire(main)
