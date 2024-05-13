from scripts.train_phi_sae import main

layers = [8, 12, 16, 20, 24, 28]
# for layer_idx in range(3, len(layers)):
for layer_idx in [0]:
    layer = layers[layer_idx]
    sc = 1.5e-6 + 4e-7 * (layer - 10)  # + 0.5e-7 * max(layer - 12, 0)
    main(layer=layer, is_gated=True, sparsity_coefficient=sc, n_devices=4, use_recip=True, 
        #  death_penalty_threshold=9e-5,
         death_penalty_threshold=5e-5,
         train_steps=10_000, push_to_hub=("nev/phi-3-4k-saex-test", f"l{layer}-test-run-1"),
         restore=f"weights/phi-l{layers[layer_idx-1]}-gated.safetensors" if layer_idx > 0 else None,
         ema=0.999
         )

# nohup poetry run python train_phi_sae.py &
