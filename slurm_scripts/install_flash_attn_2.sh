#!/bin/bash
#SBATCH --job-name=fa_cu126_clean
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

module purge
module load cuda/12.6
module load gcc11/11.3.0   # or gcc/11.2.0 or gcc12/12.2.0 if it really provides a different g++

# IMPORTANT: do NOT 'module load anaconda3' (it pulls cuda/11.8)
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export PYTHONNOUSERSITE=1
unset PYTHONPATH

# Force correct CUDA
export CUDA_HOME=/gpfs/share/apps/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

# Force compiler that pip/torch extension build uses
export CC=$(which gcc)
export CXX=$(which g++)
export MAX_JOBS=4
export TORCH_CUDA_ARCH_LIST="8.0"

echo "=== Debug: python ==="
python -c "import sys; print(sys.executable)"
echo "=== Debug: gcc/g++ ==="
which gcc; gcc --version | head -n 1
which g++; g++ --version | head -n 1
echo "=== Debug: nvcc ==="
which nvcc; nvcc --version | tail -n 1
echo "=== Debug: torch ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"

# Build deps
python -m pip install -U pip setuptools wheel
python -m pip install -U ninja cmake

# Remove old broken installs
python -m pip uninstall -y flash-attn || true

# Install from source against your torch/cu126 + CUDA 12.6
python -m pip install --no-build-isolation --no-cache-dir flash-attn

# Verify
python -c "import flash_attn; import torch; print('flash-attn OK', flash_attn.__version__, '| torch', torch.__version__)"
