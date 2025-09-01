#!/bin/bash
#SBATCH --job-name=train_modernBERT_all_uniref
#SBATCH --partition=a100_long
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=28-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err

# srun --job-name=train_bert --partition=a100_long --gres=gpu:a100:1 --cpus-per-task=16 --mem=100G --time=28-00:00:00
# srun --job-name=train_bert_ddp --partition=a100_dev --gres=gpu:a100:2 --cpus-per-task=16 --mem=100G --time=04:00:00 --pty /bin/bash

# === Load and activate conda environment ===
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Print environment info for reproducibility ===
echo "Python executable: $(which python)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
nvidia-smi

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === Weights & Biases config ===
export WANDB_PROJECT=huggingface_bert_sweep
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762
# f46013ff59ad00162f0adb0eb8dd03811505fbc6
# === Change to project directory ===
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# === Use a random master port for torch distributed ===
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"
# export MASTER_PORT=12345
# === Run training using torchrun with 2 processes ===
# torchrun --nproc_per_node=2 --master_port=$MASTER_PORT python_scripts/multi_gpu_train.py
# torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/working_train2.py
# torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/modernBERT_single_gpu.py
torchrun --nproc_per_node=1 --master_port=$MASTER_PORT python_scripts/modernBERT_long_ctxt_length.py