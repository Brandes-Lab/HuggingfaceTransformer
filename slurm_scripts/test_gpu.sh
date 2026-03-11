#!/bin/bash
#SBATCH --job-name=T5Gemma_97M_phylo_bs_4096_arrow_lmdb
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=03-00:00:00
#SBATCH --output=T5Gemma_97M_phylo_bs_4096_arrow_lmdb_%j.out
#SBATCH --error=T5Gemma_97M_phylo_bs_4096_arrow_lmdb_%j.err

set -euo pipefail

# --- environment ---
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

echo "which python: $(which python)"
echo "which torchrun: $(which torchrun)"
python -c "import sys; print(sys.executable)"

# If you previously hit cuDNN mismatches, keep this; otherwise you can remove it.
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v '^/gpfs/share/apps/cuda/12.6' \
  | paste -sd: -)

# --- distributed basics (single node, single proc) ---
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=$((12000 + RANDOM % 20000))

# --- caches / logging ---
export HF_HOME=/gpfs/data/brandeslab/User/as12267/cache/huggingface
export TOKENIZERS_PARALLELISM=false

# --- run ---
cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/


# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_mlm_6_lr3e-4_bs4096_ctxt_1024 \
#   --model_type "ModernBERT" \
#   --training_type "MLM" \
#   --mlm_probability 0.06 \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "seq_pair_map" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/home/as12267/train.jsonl \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --vep_eval_steps 3742 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "no" \
#   --save_strategy "epoch"


# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name T5Gemma_97M_phylo_lr1e-4_bs256_ctxt_1024_arrow_dataset_lmdb \
#   --model_type "T5Gemma" \
#   --training_type "phylo_encoder_decoder" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "uniref90_arrow" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_bk.lmdb \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 8 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 1e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "no" \
#   --save_strategy "epoch"



# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name T5Gemma_97M_phylo_bs_4096_arrow_fasta_file \
#   --model_type "T5Gemma" \
#   --training_type "phylo_encoder_decoder" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "uniref90_arrow_fasta" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 1e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "steps" \
#   --eval_steps 500 \
#   --per_device_eval_batch_size 256 \
#   --save_steps 500 \
#   --save_strategy "steps"


# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_phylo_lr3e-4_bs4096_ctxt_1024_arrow_dataset_index_file \
#   --model_type "ModernBERT" \
#   --training_type "phylo_encoder_only" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "uniref90_arrow_fasta" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --vep_eval_steps 500 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "steps" \
#   --eval_steps 500 \
#   --per_device_eval_batch_size 256 \
#   --save_steps 500 \
#   --save_strategy "steps"



# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_mlm_lr3e-4_bs4096_ctxt_1024_arrow_dataset_index_file \
#   --model_type "ModernBERT" \
#   --training_type "MLM" \
#   --mlm_probability 0.06 \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "uniref90_arrow_fasta" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --vep_eval_steps 500 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 16 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 8 \
#   --eval_strategy "steps" \
#   --eval_steps 500 \
#   --per_device_eval_batch_size 256 \
#   --save_steps 500 \
#   --save_strategy "steps"


# torchrun \
#   --nproc-per-node=1 \
#   --master_addr="${MASTER_ADDR}" \
#   --master_port="${MASTER_PORT}" \
#   python_scripts/train_modernBERT.py \
#   --run-name modernBERT_113M_phylo_lr3e-4_bs4096_ctxt_1024_arrow_lmdb \
#   --model_type "ModernBERT" \
#   --training_type "phylo_encoder_only" \
#   --wandb_project "phylo-llm" \
#   --tokenizer-path ./phylo_char_tokenizer_updated \
#   --train_dataset_type "uniref90_arrow_fasta" \
#   --max_position_embeddings 1024 \
#   --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
#   --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
#   --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
#   --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
#   --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
#   --output-dir /gpfs/data/brandeslab/model_checkpts \
#   --num_train_epochs 100 \
#   --vep_eval_steps 500 \
#   --logging_steps 4 \
#   --batch_sampler "phylo_default" \
#   --per_device_train_batch_size 128 \
#   --gradient_accumulation_steps 32 \
#   --learning_rate 3e-4 \
#   --dataloader_num_workers 4 \
#   --dataloader_persistent_workers True \
#   --dataloader_prefetch_factor 2 \
#   --eval_strategy "steps" \
#   --eval_steps 1 \
#   --per_device_eval_batch_size 256 \
#   --save_steps 500 \
#   --save_strategy "steps"  


torchrun \
  --nproc-per-node=1 \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  python_scripts/train_modernBERT.py \
  --run-name T5Gemma_97M_phylo_bs_4096_arrow_lmdb \
  --model_type "T5Gemma" \
  --training_type "phylo_encoder_decoder" \
  --wandb_project "phylo-llm" \
  --tokenizer-path ./phylo_char_tokenizer_updated \
  --train_dataset_type "uniref90_arrow_lmdb" \
  --max_position_embeddings 1024 \
  --train_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/train \
  --val_dataset_path /gpfs/data/brandeslab/Data/uniref/uniref90_clusters_arrow/test \
  --lmdb_path /gpfs/data/brandeslab/Data/uniref/uniref100_merged.lmdb \
  --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output-dir /gpfs/data/brandeslab/model_checkpts \
  --num_train_epochs 100 \
  --logging_steps 4 \
  --batch_sampler "phylo_default" \
  --per_device_train_batch_size 128 \
  --gradient_accumulation_steps 32 \
  --learning_rate 1e-4 \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 8 \
  --eval_strategy "steps" \
  --eval_steps 500 \
  --per_device_eval_batch_size 256 \
  --save_steps 500 \
  --save_strategy "steps"
