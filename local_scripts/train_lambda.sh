export WANDB_PROJECT=huggingface_bert_sweep
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762


python python_scripts/train_modernBERT.py \
    --run-name modernBERT-dynamic-batch-test-bucketed \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /home/ubuntu/filesystem1/data/train_representative \
    --val-dataset-path /home/ubuntu/filesystem1/data/uniref90_tokenized_8192_small/validation \
    --vep-input-csv /home/ubuntu/filesystem1/data/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --output-dir ./checkpoints \
    --max-steps 2000000 \
    --gradient-accumulation-steps 32 \
    --dynamic-batching \
    --group-by-length=false