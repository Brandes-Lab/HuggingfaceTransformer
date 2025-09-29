# Simple Makefile for HuggingFace Transformer Project

.PHONY: env env-uv install install-uv clean clean-uv

uv:
	curl -LsSf https://astral.sh/uv/install.sh | sh

# Create conda environment
env:
	conda env create -f environment.yml

# Create uv environment
env-uv: uv
	uv venv .venv --python 3.10
	uv pip install -r requirements.txt

# Install/update conda environment
install:
	conda env update -f environment.yml --prune
	pip install --no-deps -e .

# Install/update uv environment
install-uv: uv
	uv pip install -r requirements.txt
	uv pip install --no-deps -e .

# Remove conda environment
clean:
	conda env remove -n huggingface_bert -y

# Remove uv environment
clean-uv:
	rm -rf .venv
