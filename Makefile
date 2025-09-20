.PHONY: help install install-dev install-uv install-uv-dev install-uv-req install-uv-req-dev install-conda install-conda-uv clean test lint format check-format type-check all-checks
.DEFAULT_GOAL := help

PYTHON := python3
PIP := pip3
CONDA := conda
UV := uv

help: ## Show this help message
	@echo "Available targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package and dependencies
	$(PIP) install -e .

install-dev: ## Install the package with development dependencies
	$(PIP) install -e ".[dev]"

install-uv: ## Install the package and dependencies using uv
	$(UV) pip install -e .

install-uv-dev: ## Install the package with development dependencies using uv
	$(UV) pip install -e ".[dev]"

install-uv-req: ## Install using requirements.txt with uv
	$(UV) pip install -r requirements.txt
	$(UV) pip install -e .

install-uv-req-dev: ## Install using requirements-dev.txt with uv
	$(UV) pip install -r requirements-dev.txt
	$(UV) pip install -e .

install-conda: ## Create and setup conda environment
	$(CONDA) env create -f environment.yml || $(CONDA) env update -f environment.yml
	@echo "Activate the environment with: conda activate huggingface_bert"

install-conda-uv: ## Create conda environment and install with uv
	$(CONDA) env create -f environment.yml || $(CONDA) env update -f environment.yml
	@echo "Environment created. Now activate it and install with uv:"
	@echo "  conda activate huggingface_bert"
	@echo "  make install-uv-dev"

clean: ## Clean up build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ -v --cov=genelm --cov-report=html --cov-report=term

lint: ## Run linting checks
	flake8 genelm/
	isort --check-only genelm/
	black --check genelm/

format: ## Format code
	isort genelm/
	black genelm/

check-format: ## Check if code is properly formatted
	black --check genelm/
	isort --check-only genelm/

type-check: ## Run type checking
	mypy genelm/

all-checks: lint type-check test ## Run all checks (lint, type-check, test)

build-tokenizer: ## Build character tokenizer
	$(PYTHON) -m genelm.build_char_tokenizer

tokenize-dataset: ## Tokenize dataset (requires DATASET_PATH)
	@if [ -z "$(DATASET_PATH)" ]; then \
		echo "Error: DATASET_PATH is required. Usage: make tokenize-dataset DATASET_PATH=/path/to/dataset"; \
		exit 1; \
	fi
	$(PYTHON) -m genelm.tokenize_dataset --dataset_path $(DATASET_PATH)

train-single-gpu: ## Train model on single GPU
	$(PYTHON) -m genelm.modernBERT_single_gpu

train-multi-gpu: ## Train model on multiple GPUs (requires NPROC)
	@if [ -z "$(NPROC)" ]; then \
		echo "Error: NPROC is required. Usage: make train-multi-gpu NPROC=2"; \
		exit 1; \
	fi
	torchrun --nproc_per_node=$(NPROC) -m genelm.multi_gpu_train

# Development shortcuts
dev-setup: install-dev build-tokenizer ## Complete development setup
dev-setup-uv: install-uv-dev build-tokenizer ## Complete development setup with uv

# SLURM job shortcuts (for HPC environments)
submit-train: ## Submit training job to SLURM
	sbatch slurm_scripts/train_bert.sh

submit-tokenize: ## Submit tokenization job to SLURM
	sbatch slurm_scripts/tokenize_dataset.sh
