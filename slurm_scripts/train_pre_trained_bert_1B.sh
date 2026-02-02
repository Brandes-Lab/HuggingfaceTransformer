#!/bin/bash
#SBATCH --job-name=train_pre_trained_modernBERT_1B_3
#SBATCH --partition=reservation
#SBATCH --reservation=brandeslab_reservation
#SBATCH --nodelist=a100nv-4005,a100nv-4006,a100nv-4007
#SBATCH --gres=gpu:4
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=13-00:00:00
#SBATCH --output=/gpfs/scratch/an4477/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/an4477/slurm_logs/%x_%j.err


# === Load and activate conda environment ===
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Print environment info for reproducibility ===
echo "=============================================="
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NODELIST"
echo "Num nodes: $SLURM_NNODES"
echo "=============================================="

echo "Python executable: $(which python)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
nvidia-smi

# === Multi-node configuration ===
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

# Find a free port
while true; do
  PORT=$(shuf -i 29500-29999 -n 1)
  netstat -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT

NNODES=$SLURM_NNODES
GPUS_PER_NODE=4
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === NCCL settings for multi-node ===
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_NSOCKS_PERTHREAD=4
export NCCL_SOCKET_NTHREADS=2

# Prevent memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# === Weights & Biases config ===
export WANDB_PROJECT=long_runs
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762
export TOKENIZERS_PARALLELISM=false


# === Change to project directory ===
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# === Launch with srun + torchrun for multi-node ===
srun --kill-on-bad-exit=1 torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    python_scripts/train_pre_trained_modernBERT.py \
    --run-name pre_trained_modernBERT_1B_3 \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/train_only/train \
    --val-dataset-path /gpfs/data/brandeslab/Data/processed_datasets/uniref90_tokenized_8192/val_only/validation \
    --ckpt_path /gpfs/data/brandeslab/model_checkpts/pre_trained_modernBERT_1B_2/checkpoint-46500 \
    --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
    --vep_eval_steps 24000 \
    --output-dir /gpfs/data/brandeslab/model_checkpts \
    --max_position_embeddings 8192 \
    --max-steps 1_000_000 \
    --attn-implementation flash_attention_2 \
    --batch-sampler token_budget \
    --max-tokens-per-batch 25000 \
    --gradient-accumulation-steps 1 \
    --learning_rate 1e-4 \
    --dataloader_num_workers 4 \
    --dataloader_persistent_workers True \
    --dataloader_prefetch_factor 2 \
    --eval_strategy "no" \
    --save_steps 12000

echo "=============================================="
echo "Job finished: $(date)"
echo "=============================================="
