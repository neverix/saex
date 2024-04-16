# saex
Sparse autoencoders in Jax.

## Running
Toy model:
```bash
JAX_PLATFORMS=cpu python -m saex.toy_models
```

Tests:
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
