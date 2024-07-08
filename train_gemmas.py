import os
# layers = [3, 4, 7, 8, 9][3:]
layers = [12, 14, 16, 11, 13, 15, 10, 6, 8, 9, 7, 5]
for layer_idx in range(len(layers)):
    layer = layers[layer_idx]
    restore = None  # if layer_idx == 0 else f"weights/phi-l{layers[layer_idx-1]}-gated.safetensors"
    # fn = lambda x: x * ((layer / 12) ** 2)
    fn = lambda x: x
    for s, sae_type in ((2e-5, "residual"), (2e-5, "attn_out"))[:1]:
        min_sfc, max_sfc = fn(s), fn(s)
        # min_sfc, max_sfc = fn(1e-5), fn(1e-5)
        min_sfc, max_sfc = min_sfc, min_sfc
        os.system(f'python -m scripts.train_gemma_sae --layer {layer} --restore "{restore}" '
                f'--min_sfc {min_sfc} --max_sfc {max_sfc} --n_train=1 --sae_type="{sae_type}"')

# nohup poetry run python train_phis.py &
