import os
layers = [12]
for layer_idx in range(len(layers)):
    layer = layers[layer_idx]
    restore = None
    fn = lambda x: x
    cf = 1
    s, sae_type = 6e-5, "residual"
    for use_8bit in (False, True):
        min_sfc, max_sfc = fn(s), fn(s)
        min_sfc, max_sfc = min_sfc, min_sfc
        os.system(f'python -m scripts.train_gemma2_sae --layer {layer} --restore "{restore}" '
                f'--min_sfc {min_sfc} --max_sfc {max_sfc} --n_train=1 --sae_type="{sae_type}" '
                + (f'--use_8bit' if use_8bit else ''))