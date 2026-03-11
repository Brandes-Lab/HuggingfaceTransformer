#!/bin/bash
#SBATCH --job-name=T5Gemma_PLL_bs_4096
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50G
#SBATCH --time=03-00:00:00
#SBATCH --output=logs/T5Gemma_PLL_bs_4096_%A_%a.out
#SBATCH --error=logs/T5Gemma_PLL_bs_4096_%A_%a.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

# If you previously hit cuDNN mismatches, keep this; otherwise you can remove it.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

# --- distributed basics (single node, single proc) ---
export MASTER_PORT=$((12000 + RANDOM % 20000))

# --- caches / logging ---
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

# === Confirm CUDA + PyTorch ===
echo "Python executable: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# === Additional CUDA diagnostics ===
python -c "import torch; print('CUDA version (compiled):', torch.version.cuda)"
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"


# (optional but common)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


CKPT_LIST="checkpoints.txt"
CKPT_PATH=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "${CKPT_LIST}")

echo "Task ${SLURM_ARRAY_TASK_ID} using checkpoint: ${CKPT_PATH}"

torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/pll_new.py \
  --model_ckpt "${CKPT_PATH}" \
  --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --max_len 1024 \
  --batch_size 16 \
  --pll_mode wtenc \
  --run_name "pll_$(basename "$(dirname "${CKPT_PATH}")")_$(basename "${CKPT_PATH}")" \
  --out_dir /gpfs/data/brandeslab/User/as12267/T5Gemma_PLL_bs_4096_results