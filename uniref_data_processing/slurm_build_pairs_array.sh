#!/bin/bash
#SBATCH --partition a100_short
#SBATCH --nodes 1
#SBATCH --mem 64G
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 2
#SBATCH --time 1-12:00:00
#SBATCH --job-name build_pairs
#SBATCH --output build_pairs-%A_%a.log
#SBATCH --array=0-356%8

set -euo pipefail

module load miniconda3/gpu/4.9.2
module add cuda/11.8
source /gpfs/share/apps/miniconda3/gpu/4.9.2/etc/profile.d/conda.sh
conda activate rl_trading

unset PYTHONPATH
unset PYTHONNOUSERSITE

SHARDS_DIR="/gpfs/data/brandeslab/Data/uniref/shards_uniref90_ge2"
INDEX_DB="/gpfs/data/brandeslab/Data/uniref/uniref100_bk_test.idx.sqlite"
OUT_BASE="/gpfs/data/brandeslab/Data/uniref/hf_pairs_uniref90_sharded"
P_SELECT="0.02"
SEED="42"

mapfile -t SHARDS < <(ls -1 ${SHARDS_DIR}/shard_*.tsv | sort)

if [ "$SLURM_ARRAY_TASK_ID" -ge "${#SHARDS[@]}" ]; then
  echo "ERROR: SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID but only ${#SHARDS[@]} shards found in ${SHARDS_DIR}"
  exit 1
fi

SHARD_TSV="${SHARDS[$SLURM_ARRAY_TASK_ID]}"
echo "Shard: $SHARD_TSV"

python build_pairs_worker.py \
  --shard_tsv "$SHARD_TSV" \
  --index_db "$INDEX_DB" \
  --out_dir "$OUT_BASE" \
  --seed "$SEED" \
  --p_select "$P_SELECT" \
  --use_hashed_pair_keys