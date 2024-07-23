from typing import Optional

import fire
import jax
import jax.numpy as jnp
import numpy as np

from saex.models.micrlhf_model import MicrlhfModelConfig
from saex.train_script import train_main
from saex.trainer_cache import (BufferTrainerConfig, IterableDatasetConfig,
                                SAEConfig)


def train(
    train_steps: int = 100_000,
    n_devices: int = 4,
    mp_devices: int = 1,
    cache_size = 2**16,
    cache_batch_size = 256,
    cache_ratio=1.0,
    batch_size = 2048,
    max_seq_len = 128,
    sparsity_coefficients=[4e-6],
    # save_steps=2500,
    save_steps=250_000,
    eval_loss_every=100,
    restore = False,
    wandb_entity = "neverix",
    sae_type = "residual",
    layer = 12,
    is_gated: bool = True,
    use_recip=False,
    death_penalty_threshold=9e-5,
    push_to_hub=None,
    ema=None,
):
    n_features = 2048

    configs = []
    for sparsity_coefficient in sparsity_coefficients:
        config = BufferTrainerConfig(
            n_dimensions=n_features,
            lr=4e-4,  # Higher LR limits the variance explained but can lead to faster decrease in L0
            beta1=0.0,  # Crucial for avoiding dead features
            beta2=0.99,  # Lower beta2 can lead to a slightly faster decrease in L0
            scheduler_warmup=128,
            scheduler_cycle=max(100_000, train_steps * 2),
            scheduler_multiply=0.1,
            train_iterations=train_steps,
            save_steps=save_steps,
            use_wandb=(wandb_entity, "saex") if wandb_entity else None,
            log_every=25,
            hist_every=100,
            save_path=f"weights/gemma-l{layer}{'-gated' if is_gated else ''}.safetensors",
            dry_run_steps=0,
            no_update=False,
            sae_config=SAEConfig(
                n_dimensions=n_features,
                sparsity_loss_type="recip" if use_recip else "l1",
                recip_schedule = ((1e10, 0.1),),
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
                death_loss_type="dm_ghost_grads",
                anthropic_norm=True,
                # norm_input="wes-clip",
                # norm_input="wes",
                # norm_input="wes-mean",
                norm_input="wes-mean-fixed",
                # wes_clip=(0.25, 0.25),
                death_penalty_threshold=death_penalty_threshold,
                death_penalty_coefficient=0.25,
                dead_after=2_000,
                buffer_size=2_000,
                sparsity_tracking_epsilon=0.1,
                is_gated=is_gated,
                param_dtype="bfloat16",
                bias_dtype="float32",
                # bias_dtype="bfloat16",
                misc_dtype="bfloat16",
                # misc_dtype="float32",
                # param_dtype="float16",
                # param_dtype="float32",
                restrict_dec_norm=None,
                project_grads_from_dec=False,
                project_updates_from_dec=False,
                weights_8bit=False,
                # weights_8bit=True,
                use_aqt=False,
                topk_k=None,
                # topk_k=64,
                # topk_approx=True,
                topk_approx=False,
            ),
            # optimizer="adafactor",
            optimizer="adam",
            # optimizer="adam8",
            sae_restore=restore,
            cache_every_steps=int(cache_size / batch_size * cache_ratio),
            cache_batch_size=cache_batch_size,
            cache_acc=int(cache_size / cache_batch_size / max_seq_len),
            buffer_max_samples=cache_size,
            restore_buffer=False,
            save_buffer=False,
            model_config=MicrlhfModelConfig(
                tokenizer_path="alpindale/gemma-2b",
                # gguf_path="weights/gemma-2b.gguf",
                gguf_path="../micrlhf-progress/models/gemma-2b-it.gguf",
                device_map=f"auto:mp={mp_devices}" if n_devices > 1 else "tpu:0",
                use_flash=False,
                layer=layer,
                max_seq_len=max_seq_len,
                from_type="gemma",
                load_eager=True,
                sae_type=sae_type,
            ),
            dataset_config=IterableDatasetConfig(
                dataset_name="HuggingFaceFW/fineweb",
            ),
            loss_batch_size=16,
            eval_loss_every=eval_loss_every,
            buffer_dtype="bfloat16",
            # buffer_dtype="float16",
            # buffer_dtype="float32",
            use_devices=n_devices,
            mp_devices=mp_devices,
            push_to_hub=None if push_to_hub is None else (push_to_hub[0], push_to_hub[1]
                                                          + f"-{sparsity_coefficient:.2E}"),
            ema=ema,
        )
        configs.append(config)
    train_main(configs)


def main(layer: int = 12, restore: Optional[str] = None, min_sfc=2e-5, max_sfc=5e-5, n_train=4, sae_type="residual"):
    sfcs = np.linspace(min_sfc, max_sfc, n_train)
    is_recip = False
    is_gated = True
    train(layer=layer, is_gated=is_gated,
          sparsity_coefficients=sfcs,
          n_devices=4, use_recip=is_recip,
        #   death_penalty_threshold="auto",
          death_penalty_threshold=5e-6,  # <= 70 (L0) / 90k (features)
          train_steps=50_000,
        #   push_to_hub=("nev/gemma-2b-saex-test", f"l{layer}-{sae_type}-test-run-6"),
          push_to_hub=("nev/gemma-2b-saex-test", f"it-l{layer}-{sae_type}-test-run-1"),
          restore=restore,
          sae_type=sae_type,
          )


if __name__ == "__main__":
    fire.Fire(main)
