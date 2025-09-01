#!/bin/bash
#SBATCH --job-name=tok_val
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out

# Load Anaconda module
module load anaconda3

# Activate your Hugging Face environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# Optional: set number of threads for dataset.map
# export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

python python_scripts/tokenize_uniref.py --split val
