# saex
Sparse autoencoders in Jax.

## Running
```bash
# Train a small SAE on the GPT-2 residual stream. Requires at most 32GB of RAM.
poetry run python -m saex.trainer_cache
# Download GPT-2 residual stream SAEs for finetuning
scripts/download_jb_saes.sh
# Generate data for a toy model
JAX_PLATFORMS=cpu python -m saex.toy_models
```

Tests (there aren't any yet):
```bash
poetry run pytest
```

## How to install
```bash
sudo apt install -y make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libffi-dev liblzma-dev python-openssl git
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n eval "$(pyenv init -)"\nfi' >> ~/.bashrc
pyenv install 3.12.3
pyenv global 3.12.3
python3 -m pip install poetry
echo 'export PATH="$PYENV_ROOT/versions/3.12.3/bin:$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

poetry env use 3.12
poetry lock
poetry install
poetry shell
```
I think it should be possible to set up a development environment without installing pyenv on tpu-ubuntu2204. 

# FAQ
No one actually asked these questions, but here are the answers anyway.

> How is this code parallelized?

Data and tensor parallelism. In theory, the size of the SAE is unlimited. In practice, it is initialized on one device.

> Are results comparable to SAELens?

Yes. I haven't tested with smaller batch sizes, but you can get comparable results for GPT2-Small Layer 9 with ~25% less tokens and ~3x lower training time.

> What techniques does `saex` use?

* [Methodology is overall similar to Bloom 2024](https://www.lesswrong.com/posts/f9EgfLSurAiqRJySD/open-source-sparse-autoencoders-for-all-residual-stream)
* [Decoder weight parametrization from Wright 2024](https://www.lesswrong.com/posts/3JuSjTZyMzaSeTxKk/addressing-feature-suppression-in-saes)
* [An alternative L0 approximation by Eoin Farrell](https://www.lesswrong.com/posts/cYA3ePxy8JQ8ajo8B/experiments-with-an-alternative-method-to-promote-sparsity)
* [Deepmind's scaled ghost gradients, modified to use Softplus](https://www.alignmentforum.org/posts/C5KAZQib3bzzpeyrg/progress-update-1-from-the-gdm-mech-interp-team-full-update)

# TODOs
* LLaMA/TinyLLaMA support
* [Gated SAEs](https://arxiv.org/abs/2404.16014)
* Make loss evaluation faster (jitted)
* Tensor parallel for models
