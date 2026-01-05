#!/bin/bash -l
#SBATCH --job-name=train_u-net
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err

# === Load modules ===
module load anaconda3/gpu/new   # match what worked for as12267

# === Activate environment ===
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Cache location (use your own space) ===
export HF_HOME=/gpfs/scratch/an4477/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
mkdir -p $HF_HOME

echo "Caching to: $HF_HOME"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# === Navigate to project ===
cd /gpfs/data/brandeslab/HuggingfaceTransformer/python_scripts

export TORCHINDUCTOR_DISABLE_CUDA_GRAPH=1
export TORCH_COMPILE_DEBUG=0

# === Run training ===
python train_modernBERT.py

