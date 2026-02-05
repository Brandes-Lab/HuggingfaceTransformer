#!/bin/bash
#SBATCH --job-name=t5_small_phylo_lr_1e-3_bs_4096_in_memory_dataset
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err


# === Load and activate conda environment ===
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

# === Print environment info for reproducibility ===
echo "Python executable: $(which python)"
python -c "import transformers; print('Transformers:', transformers.__version__)"
nvidia-smi

while true; do
  PORT=$(shuf -i 20000-25000 -n 1)
  netstat -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT
echo "Selected free MASTER_PORT: $PORT"

export WORLD_SIZE=$(($SLURM_NNODES*2))
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP:" $head_node_ip

# Set Hugging Face cache location to non-home directory
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
echo "Caching to: $HF_HOME"

# === Weights & Biases config ===
export WANDB_PROJECT=long_runs
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export TOKENIZERS_PARALLELISM=false

# === Change to project directory ===
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# === Use a random master port for torch distributed ===
export MASTER_PORT=$((29500 + RANDOM % 1000))

# torchrun \
#     --nnodes=1 \
#     --nproc-per-node=1 \
#     --master_addr=${MASTER_ADDR} \
#     --master_port=${MASTER_PORT} \
#     --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
#     --rdzv_backend=c10d \
#     python_scripts/train_modernBERT.py \
#     --run-name modernBERT_34M_phylo \
#     --tokenizer-path ./phylo_char_tokenizer \
#     --train-dataset-path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters.parquet \
#     --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#     --output-dir /gpfs/data/brandeslab/model_checkpts \
#     --max-steps 3_000_000 \
#     --batch_sampler "default" \
#     --per_device_train_batch_size 8 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 1e-3 \
#     --vep_eval_steps 5_000 \
#     --dataloader_num_workers 6 \
#     --dataloader_persistent_workers True \
#     --dataloader_prefetch_factor 2 \
#     --eval_strategy "no" \
#     --save_steps 5_000 \
    
# torchrun \
#   --nnodes=1 \
#   --nproc-per-node=1 \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
#   --rdzv_backend=c10d \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_34M_phylo_lr_3e-3 \
#   --training_type "phylo" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train-dataset-path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters.parquet \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --max-steps 3_000_000 \
#   --logging_steps 16 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 3e-3 \
#   --vep_eval_steps 1_000 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 4 \
#   --eval_strategy "no" \
#   --save_strategy "no" 


torchrun \
  --nnodes=1 \
  --nproc-per-node=1 \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
  --rdzv_backend=c10d \
  python_scripts/train_modernBERT.py \
  --run-name modernBERT_34M_phylo_lr_6e-4_bs_4096_in_memory_dataset \
  --model_type "ModernBERT" \
  --training_type "phylo_encoder_only" \
  --wandb_project "phylo-llm" \
  --tokenizer-path ./phylo_char_tokenizer_updated \
  --train_dataset_type "iterable" \
  --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
  --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
  --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
  --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output-dir /gpfs/data/brandeslab/model_checkpts \
  --max-steps 3_000_000 \
  --logging_steps 16 \
  --batch_sampler "phylo_default" \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 32 \
  --learning_rate 6e-4 \
  --vep_eval_steps 230 \
  --dataloader_num_workers 32 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 16 \
  --eval_strategy "no" \
  --save_strategy "no" 


# time torchrun \
#   --nnodes=1 \
#   --nproc-per-node=1 \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
#   --rdzv_backend=c10d \
#   python_scripts/train_modernBERT.py \
#   --run-name t5_small_phylo_lr_1e-3_bs_4096_in_memory_dataset \
#   --model_type "T5" \
#   --training_type "phylo_encoder_decoder" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "iterable" \
#   --max_position_embeddings 512 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --max_steps 10_000_000 \
#   --logging_steps 8 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 1e-3 \
#   --vep_eval_steps 230 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "no" \
#   --save_strategy "no" 