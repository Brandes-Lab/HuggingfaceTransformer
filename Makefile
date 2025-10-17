# Simple Makefile for HuggingFace Transformer Project

.PHONY: env env-uv install install-uv clean clean-uv

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Create conda environment
env:
	conda env create -f environment.yml

# Create uv environment
env-uv: install-uv
	uv venv .venv --python 3.10 && \
	source .venv/bin/activate && \
	uv pip install -e .[training] setuptools

# Install/update conda environment
install-env:
	conda env update -f environment.yml --prune

# Install/update uv environment
install-env-uv:
	uv pip install -r requirements.txt

# Remove conda environment
clean:
	conda env remove -n huggingface_bert -y

# Remove uv environment
clean-uv:
	rm -rf .venv

lambda-gh-auth:
	sudo apt install gh && \
	gh auth login

lambda-git-config:
	git config --global user.name "Benjamin Levy" && \
	git config --global user.email "benjaminjslevy@gmail.com" && \
	git config push.autoSetupRemote true

lambda-rclone-from-local:
	rclone copy . lambda:/home/ubuntu/filesystem3/HuggingfaceTransformer -P  --exclude ".venv/**" --exclude "__pycache__/**" --exclude "*.pyc" --exclude ".mypy_cache/**" --exclude ".git/**" --exclude "wandb/**" --exclude "*.pt.trace.json" --exclude "checkpoints/**";
	rclone copy ../../data lambda:/home/ubuntu/filesystem3/data -P  --exclude ".venv/**" --exclude "__pycache__/**" --exclude "*.pyc" --exclude ".mypy_cache/**" --exclude "*.pt.trace.json";

install-flash-attn:
	uv pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.17/flash_attn-2.8.3+cu128torch2.9-cp310-cp310-linux_x86_64.whl