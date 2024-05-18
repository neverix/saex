from scripts.train_phi_sae import main
import os
layers = [12, 16, 14, 18]
for layer_idx in range(len(layers)):
    layer = layers[layer_idx]
    restore = None  # if layer_idx == 0 else f"weights/phi-l{layers[layer_idx-1]}-gated.safetensors"
    fn = lambda x: x * ((layer / 12) ** 2)
    # fn = lambda x: x
    min_sfc, max_sfc = fn(2e-5), fn(5e-5)
    min_sfc, max_sfc = min_sfc, min_sfc
    os.system(f'poetry run python -m scripts.train_gemma_sae --layer {layer} --restore "{restore}" '
              f'--min_sfc {min_sfc} --max_sfc {max_sfc} --n_train=1')

# nohup poetry run python train_phis.py &
