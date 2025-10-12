#!/bin/bash
# Training script with PyTorch Profiler enabled
# This will profile the training loop and save traces to ./profiler_traces

export WANDB_PROJECT=huggingface_bert_sweep
# export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762

export WORLD_SIZE=8
echo "WORLD_SIZE="$WORLD_SIZE

export TOKENIZERS_PARALLELISM=false

# Set a random port to avoid conflicts
export MASTER_PORT=$((29500 + RANDOM % 1000))
echo "Selected MASTER_PORT: $MASTER_PORT"

# Profiling Configuration
# The profiler will:
# - Wait 5 steps (skip initial slow steps)
# - Warmup for 2 steps
# - Actively profile for 3 steps
# - Repeat the cycle 2 times
# Total profiled steps: (5 wait + 2 warmup + 3 active) * 2 = 20 steps
# Note: Profiling adds overhead, so training might be slightly slower

torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    python_scripts/train_modernBERT.py \
    --run-name modernBERT-profiling-test \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /home/ubuntu/filesystem2/uniref90_tokenized_8192_small/train \
    --val-dataset-path /home/ubuntu/filesystem2/uniref90_tokenized_8192_small/validation \
    --vep-input-csv /home/ubuntu/filesystem2/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --output-dir ./checkpoints \
    --max-steps 50 \
    --dynamic-batching \
    --enable-profiling \
    --enable-memory-logging \
    --profiler-output-dir ./profiler_traces \
    --profiler-wait-steps 5 \
    --profiler-warmup-steps 2 \
    --profiler-active-steps 3 \
    --profiler-repeat 2 \
    --memory-log-steps 5

echo ""
echo "================================================================"
echo "Profiling complete! View results with:"
echo "  tensorboard --logdir=./profiler_traces --port=6006"
echo "Then open: http://localhost:6006 and go to PYTORCH_PROFILER tab"
echo "================================================================"

