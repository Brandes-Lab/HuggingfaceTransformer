#!/bin/bash
#SBATCH --job-name=install_flash_attn
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=48
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err


# # Load modules
module load anaconda3
module load cuda/11.8  

# Activate your conda environment
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu118

export PYTHONNOUSERSITE=1
export MAX_JOBS=4
# pip install flash-attn --no-binary flash-attn --no-build-isolation --no-cache-dir
python -m pip install flash-attn --no-binary flash-attn --no-build-isolation --no-cache-dir

echo "Installation completed!"

