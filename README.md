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

Currently, we only use data parallel - splitting activation buffers and batches across devices and replicating the rest. In the future, I hope we can train giant SAEs multi-node with tensor parallel.

> Why was this code written this way?

Dependency inversion.

> What, how?

Dependency inversion.
