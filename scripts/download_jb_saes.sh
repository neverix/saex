#!/usr/bin/env bash
for layer in {0..11}
    do wget "https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/resolve/main/blocks.${layer}.hook_resid_pre/sae_weights.safetensors?download=true" -c -O "weights/jb-gpt2s-${layer}.safetensors"
done