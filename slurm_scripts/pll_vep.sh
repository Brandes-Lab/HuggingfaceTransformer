#!/bin/bash
#SBATCH --job-name=T5Gemma_97M_phylo_pll-epoch4_wt_enc
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=pll-test_epoch_4_wt_enc-%j.out
#SBATCH --error=pll-test-epoch_4_wt_enc%j.err


# === Load CUDA first ===
module load cuda/11.8

# === Verify CUDA module loaded ===
echo "CUDA_HOME: $CUDA_HOME"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
which nvcc
nvcc --version

# === Load and activate conda environment ===
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Confirm CUDA + PyTorch ===
echo "Python executable: $(which python)"
python -c "import torch; print('PyTorch version:', torch.__version__)"
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# === Additional CUDA diagnostics ===
python -c "import torch; print('CUDA version (compiled):', torch.version.cuda)"
python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"


# (optional but common)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# === MASTER PORT modification (avoid 29500 collisions) ===
export MASTER_PORT=$((12000 + RANDOM % 20000))
echo "MASTER_PORT: $MASTER_PORT"

# torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/pll_new.py \
#   --model_ckpt /gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024/checkpoint-3765 \
#   --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --max_len 1024 \
#   --batch_size 8 \
#   --pll_mode selfenc \
#   --run_name t5gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_pll_eval_epoch_1


torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/pll_new.py \
  --model_ckpt /gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024/checkpoint-15060 \
  --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --max_len 1024 \
  --batch_size 8 \
  --pll_mode wtenc \
  --run_name t5gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_pll_eval_epoch_4

# torchrun --nproc_per_node=1 python_scripts/pll.py \
#   --model_ckpt /gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024/checkpoint-15060 \
#   --zero_shot_csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --max_len 1024 \
#   --batch_size 8 \
#   --run_name t5gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_pll_eval_epoch_4



