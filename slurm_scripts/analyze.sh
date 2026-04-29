#!/bin/bash
#SBATCH --job-name=VEP_score_checkpoints
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=03:00:00
#SBATCH --output=VEP_logits_score_checkpoints_%j.out
#SBATCH --error=VEP__logits_score_checkpoints_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export PYTHONPATH=/gpfs/home/rm7569/HuggingfaceTransformer:${PYTHONPATH:-}

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

echo "Python executable: $(which python)"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
python python_scripts/obtain_logodds.py \
    --checkpoint_dir /gpfs/data/brandeslab/phylo_llm_checkpts/modernBERT_113M_prefixlm_bs512_ctxt_2048_100k_final \
    --tokenizer_path /gpfs/home/rm7569/HuggingfaceTransformer/phylo_char_tokenizer_with_bos \
    --vep_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --output_csv /gpfs/home/rm7569/HuggingfaceTransformer/clinvar_full_dist.csv \
    --steps 3400 5200 \
    --batch_size 8 \
    --max_len 2048