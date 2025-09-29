export WANDB_PROJECT=huggingface_bert_sweep
export WANDB_API_KEY=ae9049d442db2ba3fa77f7928c1dae68353b3762


python python_scripts/modernBERT_long_ctxt_length.py \
    --tokenizer-path ./char_tokenizer \
    --train-dataset-path ../data/train_representative \
    --val-dataset-path ../data/uniref90_tokenized_8192_small/validation \
    --vep-input-csv ../data/uniref90_tokenized_8192_small/clinvar_AA_zero_shot_input.csv \
    --run-name modernBERT-gpu-test \
    --output-dir ./checkpoints \
    --disable-wandb \
    --vep_eval_steps 1000
