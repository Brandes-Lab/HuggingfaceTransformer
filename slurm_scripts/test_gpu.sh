#!/bin/bash
#SBATCH --job-name=T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_2
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=test-%j.out
#SBATCH --error=test-%j.err

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

nvidia-smi

while true; do
PORT=$(shuf -i 20000-25000 -n 1)
netstat -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT
echo "Selected free MASTER_PORT: $PORT"

export WORLD_SIZE=1
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo "Head node IP:" $head_node_ip

# === HuggingFace cache setup ===
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME

# === WandB config ===
export WANDB_PROJECT=phylo-llm
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762
export TOKENIZERS_PARALLELISM=false

export MASTER_PORT=$((29500 + RANDOM % 1000))


# === Change to project dir ===
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

# === Run training ===
# torchrun \
# --nnodes=1 \
# --nproc-per-node=1 \
# --master_addr=${MASTER_ADDR} \
# --master_port=${MASTER_PORT} \
# --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
# --rdzv_backend=c10d \
# python_scripts/train_modernBERT.py \
# --run-name T5Gemma_97M_phylo_lr1e-4_bs128_ctxt_4096 \
# --model_type "T5Gemma" \
# --training_type "phylo_encoder_decoder" \
# --wandb_project "phylo-llm" \
# --tokenizer-path ./phylo_char_tokenizer_updated \
# --train_dataset_type "iterable" \
# --max_position_embeddings 4096 \
# --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
# --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
# --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
# --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
# --output-dir /gpfs/data/brandeslab/model_checkpts \
# --max_steps 10000000 \
# --logging_steps 4 \
# --batch_sampler "phylo_default" \
# --per_device_train_batch_size 4 \
# --gradient_accumulation_steps 32 \
# --learning_rate 1e-4 \
# --vep_eval_steps 1052 \
# --dataloader_num_workers 16 \
# --dataloader_persistent_workers True \
# --dataloader_prefetch_factor 8 \
# --eval_strategy "no" \
# --save_strategy "no"

# torchrun \
#   --nnodes=1 \
#   --nproc-per-node=1 \
#   --master_addr=${MASTER_ADDR} \
#   --master_port=${MASTER_PORT} \
#   --rdzv_endpoint=${head_node_ip}:${MASTER_PORT} \
#   --rdzv_backend=c10d \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_100M_phylo_lr1e-3_bs128_ctxt_4096 \
#   --training_type "phylo" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "iterable" \
#   --max_position_embeddings 4096 \
#   --train-dataset-path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters.parquet \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --max-steps 3_000_000 \
#   --logging_steps 16 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 4 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 1e-3 \
#   --vep_eval_steps 1052 \
#   --dataloader_num_workers 32 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 16 \
#   --eval_strategy "no" \
#   --save_strategy "no" 


torchrun \
--nproc-per-node=1 \
--master_port=${MASTER_PORT} \
python_scripts/train_modernBERT.py \
--run-name T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_2 \
--model_type "T5Gemma" \
--training_type "phylo_encoder_decoder" \
--wandb_project "phylo-llm" \
--tokenizer-path ./phylo_char_tokenizer_updated \
--train_dataset_type "seq_pair_map" \
--max_position_embeddings 1024 \
--train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
--index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
--fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
--vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
--output-dir /gpfs/data/brandeslab/model_checkpts \
--num_train_epochs 100 \
--logging_steps 4 \
--batch_sampler "phylo_default" \
--per_device_train_batch_size 8 \
--gradient_accumulation_steps 32 \
--learning_rate 1e-4 \
--dataloader_num_workers 16 \
--dataloader_persistent_workers True \
--dataloader_prefetch_factor 8 \
--eval_strategy "no" \
--save_strategy "steps" \
--save_steps 500 \


# torchrun \
# --nproc-per-node=1 \
# --master_port=${MASTER_PORT} \
# python_scripts/train_modernBERT.py \
# --run-name modernBERT_34M_phylo_lr6e-4_bs4096_ctxt_1024 \
# --model_type "ModernBERT" \
# --training_type "phylo_encoder_only" \
# --wandb_project "phylo-llm" \
# --tokenizer-path ./phylo_char_tokenizer_updated \
# --train_dataset_type "iterable" \
# --max_position_embeddings 1024 \
# --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
# --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
# --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
# --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
# --output-dir /gpfs/data/brandeslab/model_checkpts \
# --max_steps 10000 \
# --logging_steps 4 \
# --batch_sampler "phylo_default" \
# --per_device_train_batch_size 128 \
# --gradient_accumulation_steps 32 \
# --learning_rate 6e-4 \
# --vep_eval_steps 200 \
# --dataloader_num_workers 16 \
# --dataloader_persistent_workers True \
# --dataloader_prefetch_factor 8 \
# --eval_strategy "no" \
# --save_strategy "no"



# torchrun \
# --nproc-per-node=1 \
# --master_port=${MASTER_PORT} \
# python_scripts/train_modernBERT.py \
# --run-name modernBERT_113M_phylo_lr3e-4_bs4096_ctxt_1024 \
# --model_type "ModernBERT" \
# --training_type "phylo_encoder_only" \
# --wandb_project "phylo-llm" \
# --tokenizer-path ./phylo_char_tokenizer_updated \
# --train_dataset_type "iterable" \
# --max_position_embeddings 1024 \
# --train_dataset_path /gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_final/train.jsonl \
# --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
# --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
# --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
# --output-dir /gpfs/data/brandeslab/model_checkpts \
# --max_steps 10000 \
# --logging_steps 4 \
# --batch_sampler "phylo_default" \
# --per_device_train_batch_size 64 \
# --gradient_accumulation_steps 64 \
# --learning_rate 6e-4 \
# --vep_eval_steps 200 \
# --dataloader_num_workers 16 \
# --dataloader_persistent_workers True \
# --dataloader_prefetch_factor 8 \
# --eval_strategy "no" \
# --save_strategy "no"