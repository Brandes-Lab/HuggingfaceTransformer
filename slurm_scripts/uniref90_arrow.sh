#!/bin/bash
#SBATCH --job-name=mk_uniref90_arrow_test
#SBATCH --partition=a100_short
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03-00:00:00
#SBATCH --output=mk_uniref90_arrow_test%j.out
#SBATCH --error=mk_uniref90_arrow_test_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126


python python_scripts/make_uniref90_arrow.py --split test