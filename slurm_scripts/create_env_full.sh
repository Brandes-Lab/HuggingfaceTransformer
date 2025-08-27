#!/bin/bash
#SBATCH --job-name=create_env
#SBATCH --output=/gpfs/data/brandeslab/User/as12267/create_env_%j.log
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4

module load anaconda3

# Set up safe tmp path and conda env target
export TMPDIR=/gpfs/data/brandeslab/User/as12267/tmp
mkdir -p $TMPDIR
export CONDA_ENVS_PATH=/gpfs/data/brandeslab/User/as12267/.conda/envs

# Create the conda environment
conda create -n huggingface_bert python=3.10 -y

# Load conda shell hooks (needed for 'conda activate' to work)
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate huggingface_bert

# Install all required packages into the correct conda env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets evaluate wandb tokenizers scikit-learn pandas

