export WANDB_PROJECT=huggingface_bert_sweep
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762


python python_scripts/modernBERT_long_ctxt_length.py \
    --run-name modernBERT-dynamic-batch-test-long \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path /home/ubuntu/filesystem1/data/train_representative \
    --val-dataset-path /home/ubuntu/filesystem1/data/uniref90_tokenized_8192_small/validation \
    --vep-input-csv /home/ubuntu/filesystem1/data/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --run-name modernBERT-dynamic-batch-test \
    --output-dir ./checkpoints \
    --max-steps 2000000 \
    --gradient-accumulation-steps 4 \
    --dynamic-batching