#!/bin/bash
#SBATCH --job-name=train_unet_single_gpu
#SBATCH --partition=a100_dev
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=/gpfs/home/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/home/an4477/slurm_logs/%x_%j.err

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
export WANDB_PROJECT=unet_vep_training
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762
export TOKENIZERS_PARALLELISM=false

# === Change to project directory ===
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# === Use a random master port for torch distributed ===
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Using MASTER_PORT=$MASTER_PORT"

# === Run training on single GPU ===
torchrun \
    --nproc_per_node=1 \
    --master_port=${MASTER_PORT} \
    python_scripts/train_modernBERT.py \
    --run-name unet_single_gpu \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train \
    --val-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation \
    --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --output-dir /gpfs/home/an4477/model_checkpts \
    --max-steps 500_000 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 64 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-3 \
    --logging_steps 32 \
    --vep_eval_steps 10_000 \
    --dataloader_num_workers 6 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --eval_strategy "no" \
    --save_steps 50_000
