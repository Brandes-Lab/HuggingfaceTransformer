#!/bin/bash
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/huggingface_transformer/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/huggingface_transformer/slurm_logs/%x_%j.err

module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

cd /gpfs/data/brandeslab/Project/huggingface_transformer/
python eval_vep_over_checkpoints.py --checkpoint $1
