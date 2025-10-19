export WANDB_PROJECT=huggingface_bert_sweep
# export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export WORLD_SIZE=1
echo "WORLD_SIZE="$WORLD_SIZE

# master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# export MASTER_ADDR=$master_addr
# echo "MASTER_ADDR="$MASTER_ADDR

# nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
# nodes_array=($nodes)
# head_node=${nodes_array[0]}
# head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
# echo "Head node IP:" $head_node_ip


export TOKENIZERS_PARALLELISM=false

# Set a random port to avoid conflicts
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Selected MASTER_PORT: $MASTER_PORT"

torchrun \
    --nnodes=1 \
    --nproc-per-node=1 \
    python_scripts/train_modernBERT.py \
    --run-name modernBERT-flash-attn-token-budget \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path ../data/uniref90_tokenized_8192_small/train \
    --val-dataset-path ../data/uniref90_tokenized_8192_small/validation \
    --vep-input-csv ../data/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --output-dir ./checkpoints \
    --logging-steps 1 \
    --max-steps 48 \
    --attn-implementation flash_attention_2 \
    --batch-sampler token_budget \
    --max-tokens-per-batch 32768 \
    --gradient-accumulation-steps 16