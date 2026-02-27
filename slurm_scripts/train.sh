#!/bin/bash
#SBATCH --job-name=modernBERT_113M_mlm_6_lr3e-4_bs4096_ctxt_1024
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --exclude=gl40s-8006
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=test_%j.out
#SBATCH --error=test_%j.err

set -euo pipefail

# Start from a clean environment
module purge

# DO NOT load anaconda3/gpu/new (it forces cuda/11.8 and you can't unload it)

# Optional: if your site allows, you can load cuda/12.6; if it causes trouble, comment it out.
module load cuda/12.6 || true

# Block ~/.local packages
export PYTHONNOUSERSITE=1
unset PYTHONPATH
hash -r

# Activate conda without the anaconda module
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert

ENV_PY=/gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert/bin/python
ENV_TORCHRUN=/gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert/bin/torchrun
export PATH=/gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert/bin:$PATH
hash -r

echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi

# Hard proof CUDA init works (will crash job if not)
$ENV_PY - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("device_count:", torch.cuda.device_count())
torch.cuda.init()
print("cuda.init OK; device:", torch.cuda.get_device_name(0))
PY

# Pick a free port once
while true; do
  PORT=$(shuf -i 20000-25000 -n 1)
  netstat -tuln | grep -q ":$PORT " || break
done
export MASTER_PORT=$PORT
echo "MASTER_PORT=$MASTER_PORT"

cd /gpfs/data/brandeslab/Project/HuggingfaceTransformer/

$ENV_TORCHRUN \
  --nproc-per-node=1 \
  --master_port=$MASTER_PORT \
  python_scripts/train_modernBERT.py \
  --run-name modernBERT_113M_mlm_6_lr3e-4_bs4096_ctxt_1024 \
  --model_type "T5Gemma" \
  --training_type "MLM" \
  --mlm_probability 0.06 \
  --wandb_project "phylo-llm" \
  --tokenizer-path ./phylo_char_tokenizer_updated \
  --train_dataset_type "seq_pair_map" \
  --max_position_embeddings 1024 \
  --train_dataset_path /gpfs/home/as12267/train.jsonl \
  --index_db_path /gpfs/data/brandeslab/User/as12267/uniref100.idx \
  --fasta_path /gpfs/data/brandeslab/Data/uniref/uniref100.fasta \
  --vep-input-csv /gpfs/data/brandeslab/Data/clinvar_AA_zero_shot_input.csv \
  --output-dir /gpfs/data/brandeslab/model_checkpts \
  --num_train_epochs 100 \
  --vep_eval_steps 500 \
  --logging_steps 4 \
  --batch_sampler "phylo_default" \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 32 \
  --learning_rate 3e-4 \
  --dataloader_num_workers 16 \
  --dataloader_persistent_workers True \
  --dataloader_prefetch_factor 8 \
  --eval_strategy "no" \
  --save_strategy "epochs"
