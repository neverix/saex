import os
layers = [1, 2, 3, 4, 5]
for layer_idx in range(len(layers)):
    layer = layers[layer_idx]
    restore = None  # if layer_idx == 0 else f"weights/phi-l{layers[layer_idx-1]}-gated.safetensors"
    # fn = lambda x: x * ((layer / 12) ** 2)
    fn = lambda x: x
    # for s, sae_type in ((2e-5, "residual"), (2e-5, "attn_out"))[:1]:
    # cf = 8
    # cf = 14
    cf = 1
    # for s, sae_type in ((2e-5, "residual"), (2e-5, "attn_out")):
    for s, sae_type in ((8e-6 * cf, "transcoder"),):
        min_sfc, max_sfc = fn(s), fn(s)
        # min_sfc, max_sfc = fn(1e-5), fn(1e-5)
        min_sfc, max_sfc = min_sfc, min_sfc * 3
        os.system(f'python -m scripts.train_gemma_sae --layer {layer} --restore "{restore}" '
                f'--min_sfc {min_sfc} --max_sfc {max_sfc} --n_train=1 --sae_type="{sae_type}"')

# rm -rf wandb/ nohup.out; pkill -SIGINT -f gemma; sleep 5; nohup python train_gemmas.py & tail -F nohup.out
