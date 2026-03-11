CKPT_DIR="/gpfs/data/brandeslab/model_checkpts/T5Gemma_97M_phylo_bs_4096_arrow_dataset_index_file"

find "$CKPT_DIR" -maxdepth 1 -type d -name 'checkpoint-*' \
| awk -F'checkpoint-' '
{
  n = $2 + 0
  if (n > 1500) print $0
}' \
| sort -V > checkpoints.txt