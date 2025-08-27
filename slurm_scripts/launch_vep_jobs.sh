#!/bin/bash

CHECKPOINTS=(
  # checkpoint-100000
  # checkpoint-200000
  # checkpoint-300000
  checkpoint-400000
  # checkpoint-500000
  # checkpoint-600000
  # checkpoint-700000
  # # checkpoint-781250
)

for ckpt in "${CHECKPOINTS[@]}"; do
  sbatch --job-name=vep_${ckpt} zero_shot_vep.sh $ckpt
done
