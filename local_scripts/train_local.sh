export WANDB_PROJECT=huggingface_bert_sweep
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762


python python_scripts/modernBERT_long_ctxt_length.py \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /Users/benjaminlevy/Library/CloudStorage/OneDrive-PhysicsXLimited/Personal/Research/NYU/data/train_representative \
    --val-dataset-path /Users/benjaminlevy/Library/CloudStorage/OneDrive-PhysicsXLimited/Personal/Research/NYU/data/uniref90_tokenized_8192_small/validation \
    --vep-input-csv /Users/benjaminlevy/Library/CloudStorage/OneDrive-PhysicsXLimited/Personal/Research/NYU/data/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --run-name modernBERT-dynamic-batch-test \
    --output-dir ./checkpoints \
    --disable-wandb \
    --max-steps 100 \
