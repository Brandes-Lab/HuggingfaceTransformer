# Simple Makefile for HuggingFace Transformer Project

.PHONY: env env-uv install install-uv clean clean-uv

install-uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Create conda environment
env:
	conda env create -f environment.yml

# Create uv environment
env-uv:
	uv venv .venv --python 3.10
	uv pip install -r requirements.txt

# Install/update conda environment
install-env:
	conda env update -f environment.yml --prune

# Install/update uv environment
install-env-uv: install-uv
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
	rclone copy . lambda:/home/ubuntu/filesystem2/ -P  --exclude ".venv/**" --exclude "__pycache__/**" --exclude "*.pyc";