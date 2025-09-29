#!/bin/bash
#SBATCH --job-name=zero_shot_vep_modernBERT_34M
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=28-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err

# Load Anaconda
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === Weights & Biases config ===
export WANDB_PROJECT=modernBERT_benchmarks
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

# Move to project directory
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/


torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/zero_shot_vep.py --model_name "$1"
# sbatch slurm_scripts/zero_shot_vep.sh modernBERT_34M

