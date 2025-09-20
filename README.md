# GeneLM: Language Models for Genomic and Protein Sequences

A Python package for training BERT-style language models on biological sequences, with a focus on protein sequences and variant effect prediction.

## Features

- **Character-level tokenization** for biological sequences (amino acids, nucleotides)
- **ModernBERT architecture** with long context support (up to 8192 tokens)
- **Zero-shot variant effect prediction** evaluation during training
- **Multi-GPU training** support with PyTorch DDP
- **SLURM integration** for HPC environments
- **Weights & Biases** logging and experiment tracking

## Installation

### Quick Start

```bash
# Clone the repository
git clone https://github.com/anushkasinha/genelm.git
cd genelm

# Install the package
make install

# Or install with development dependencies
make install-dev
```

### Using uv (Fast Python Package Manager)

First, install uv if you haven't already:
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
# Or with pip: pip install uv
```

Then install the package:
```bash
# Install with uv (faster than pip)
make install-uv

# Or install with development dependencies
make install-uv-dev

# Complete development setup with uv
make dev-setup-uv

# Alternative: using requirements files
make install-uv-req      # Core dependencies
make install-uv-req-dev  # With development tools
```

### Using Conda (Recommended for HPC)

```bash
# Create conda environment
make install-conda

# Activate environment
conda activate huggingface_bert
```

### Conda + uv (Best of Both Worlds)

```bash
# Create conda environment and install with uv
make install-conda-uv

# Then activate and install
conda activate huggingface_bert
make install-uv-dev
```

### Manual Installation

```bash
# With pip
pip install -e .

# With uv (faster)
uv pip install -e .

# For development
pip install -e ".[dev]"
uv pip install -e ".[dev]"

# Using requirements files
pip install -r requirements.txt && pip install -e .
uv pip install -r requirements-dev.txt && uv pip install -e .
```

### Why Use uv?

[uv](https://github.com/astral-sh/uv) is a fast Python package manager that can significantly speed up installation:

- **10-100x faster** than pip for package installation
- **Drop-in replacement** for pip commands
- **Better dependency resolution** and conflict detection
- **Works with existing** `pyproject.toml` and `requirements.txt` files

For development workflows, uv can dramatically reduce setup time, especially when working with large ML dependencies like PyTorch and Transformers.

## Quick Start

### 1. Build a Character Tokenizer

```bash
# Build tokenizer for amino acid sequences
make build-tokenizer

# Or run directly
python -m genelm.build_char_tokenizer
```

This creates a character-level tokenizer in the `char_tokenizer/` directory with support for:
- 20 standard amino acids
- Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`
- Extended amino acid alphabet including ambiguous residues

### 2. Tokenize Your Dataset

```bash
# Tokenize a dataset (requires DATASET_PATH)
make tokenize-dataset DATASET_PATH=/path/to/your/dataset

# Or run directly
python -m genelm.tokenize_dataset --dataset_path /path/to/your/dataset
```

### 3. Train a Model

#### Single GPU Training

```bash
make train-single-gpu

# Or run directly
python -m genelm.modernBERT_single_gpu
```

#### Multi-GPU Training

```bash
make train-multi-gpu NPROC=2

# Or run directly
torchrun --nproc_per_node=2 -m genelm.multi_gpu_train
```

## Project Structure

```
genelm/
├── genelm/                     # Main Python package
│   ├── __init__.py
│   ├── build_char_tokenizer.py # Tokenizer creation
│   ├── modernBERT_long_ctxt_length.py  # Long context training
│   ├── modernBERT_single_gpu.py        # Single GPU training
│   ├── multi_gpu_train.py              # Multi-GPU training
│   ├── tokenize_dataset.py             # Dataset tokenization
│   ├── tokenize_uniref.py              # UniRef-specific tokenization
│   └── create_small_chunks.py          # Data preprocessing
├── slurm_scripts/              # SLURM job scripts
│   ├── train_bert.sh
│   ├── tokenize_dataset.sh
│   └── tokenize_train.sh
├── char_tokenizer/             # Generated tokenizer files
├── checkpoints/                # Model checkpoints
├── pyproject.toml              # Package configuration
├── Makefile                    # Build and development tasks
└── README.md
```

## Model Architecture

The package uses ModernBERT architecture with the following key features:

- **Long context support**: Up to 8192 tokens
- **Hybrid attention**: Combines global and local attention patterns
- **Configurable layers**: 8 hidden layers, 8 attention heads by default
- **Protein-optimized**: Designed for biological sequence modeling

### Default Configuration

```python
config = ModernBertConfig(
    vocab_size=32,  # Amino acids + special tokens
    max_position_embeddings=8192,
    num_hidden_layers=8,
    num_attention_heads=8,
    hidden_size=512,
    intermediate_size=2048,
    global_attn_every_n_layers=3,
    local_attention=512,
)
```

## Zero-Shot Variant Effect Prediction

The package includes built-in evaluation for variant effect prediction using ClinVar data:

```python
from genelm.modernBERT_long_ctxt_length import ZeroShotVEPEvaluationCallback

# Add to trainer
callback = ZeroShotVEPEvaluationCallback(
    tokenizer=tokenizer,
    input_csv="path/to/clinvar_data.csv",
    trainer=trainer,
    eval_every_n_steps=50000
)
trainer.add_callback(callback)
```

## SLURM Integration

For HPC environments, use the provided SLURM scripts:

```bash
# Submit training job
make submit-train

# Submit tokenization job  
make submit-tokenize

# Or submit directly
sbatch slurm_scripts/train_bert.sh
```

### Customizing SLURM Scripts

Edit the SLURM scripts in `slurm_scripts/` to match your cluster configuration:

- Update partition names
- Adjust resource requirements (GPU, CPU, memory)
- Modify conda environment paths
- Set appropriate output directories

## Development

### Code Quality

```bash
# Format code
make format

# Run linting
make lint

# Type checking
make type-check

# Run all checks
make all-checks
```

### Testing

```bash
# Run tests
make test

# Run tests with coverage
make test-cov
```

### Complete Development Setup

```bash
make dev-setup
```

## Configuration

### Environment Variables

- `WANDB_PROJECT`: Weights & Biases project name
- `WANDB_API_KEY`: Weights & Biases API key
- `HF_HOME`: Hugging Face cache directory
- `TRANSFORMERS_CACHE`: Transformers model cache
- `HF_DATASETS_CACHE`: Datasets cache directory

### Training Configuration

Key training parameters can be modified in the training scripts:

```python
training_args = TrainingArguments(
    max_steps=2_000_000,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=16,
    learning_rate=3e-4,
    bf16=True,
    group_by_length=True,
)
```

## Data Requirements

### Input Format

The package expects protein sequences in standard formats:

- **FASTA files** for sequence data
- **CSV files** for variant effect prediction with columns:
  - `sequence`: Protein sequence
  - `pos`: Position of variant (0-indexed)
  - `ref`: Reference amino acid
  - `alt`: Alternative amino acid
  - `label`: Effect label (0/1 for benign/pathogenic)

### Preprocessing

Use the provided scripts to preprocess your data:

```bash
# Create smaller chunks for efficient training
python -m genelm.create_small_chunks

# Tokenize UniRef datasets
python -m genelm.tokenize_uniref --split train
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linting (`make all-checks`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
---

**Note**: This package is under active development. APIs may change between versions.