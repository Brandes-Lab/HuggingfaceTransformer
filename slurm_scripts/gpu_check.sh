#!/bin/bash
#SBATCH --job-name=gpu_cuda_libpath
#SBATCH --partition=gl40s_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=gpu_libpath_%j.out
#SBATCH --error=gpu_libpath_%j.err

set -euo pipefail
module purge
module load cuda/12.6

source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert_cu126

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "=== locate libcuda candidates ==="
( ldconfig -p | grep libcuda || true )
( find /gpfs/share/apps/cuda/12.6 -maxdepth 4 -name 'libcuda.so*' -print 2>/dev/null || true )

echo "=== which libcuda is actually loaded? ==="
python - <<'PY'
import ctypes, os
lib = ctypes.CDLL("libcuda.so.1")
print("Loaded libcuda from:", lib._name)
PY

echo "=== torch init with CUDA toolkit paths REMOVED ==="
# Remove CUDA toolkit paths so we prefer system driver libcuda
export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | grep -v '^/gpfs/share/apps/cuda/12.6' | paste -sd: -)
echo "LD_LIBRARY_PATH(after strip)=$LD_LIBRARY_PATH"

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("is_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
