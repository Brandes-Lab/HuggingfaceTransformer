# Simple Makefile for HuggingFace Transformer Project

.PHONY: env env-uv install install-uv clean clean-uv

# Create conda environment
env:
	conda env create -f environment.yml

# Create uv environment
env-uv:
	uv venv .venv --python 3.10
	uv pip install -r requirements.txt

# Install/update conda environment
install:
	conda env update -f environment.yml --prune

# Install/update uv environment
install-uv:
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
