from scripts.train_phi_sae import main
import os
layers = [8, 12, 16, 20, 24, 28, 4, 10, 18, 26]
for layer_idx in range(len(layers)):
    layer = layers[layer_idx]
    restore = None  # if layer_idx == 0 else f"weights/phi-l{layers[layer_idx-1]}-gated.safetensors"
    fn = lambda x: x * (layer / 8) ** 2 if layer >= 8 else x
    # min_sfc, max_sfc = fn(1e-6), fn(2.5e-6)
    min_sfc, max_sfc = fn(1e-6), fn(2.5e-6)
    min_sfc, max_sfc = min_sfc, min_sfc
    os.system(f'poetry run python -m scripts.train_phi_sae --layer {layer} --restore "{restore}" '
              f'--min_sfc {min_sfc} --max_sfc {max_sfc} --n_train=1')

# nohup poetry run python train_phis.py &
