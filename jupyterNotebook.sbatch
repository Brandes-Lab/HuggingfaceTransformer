#!/bin/bash

#SBATCH --job-name=jupyterGPU
#SBATCH --partition=gl40s_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --time=03-00:00:00
#SBATCH --mem=50G
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sinha.anushka12@gmail.com
#SBATCH --output=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.out
#SBATCH --error=/gpfs/data/brandeslab/Project/slurm_logs/%x_%j.err

# ---------------------- Info ----------------------
XDG_RUNTIME_DIR=""
port=$(shuf -i8000-9999 -n1)
node=$(hostname -s)
user=$(whoami)

case "$SLURM_SUBMIT_HOST" in
  bigpurple-ln1) login_node=bigpurple1.nyumc.org ;;
  bigpurple-ln2) login_node=bigpurple2.nyumc.org ;;
  bigpurple-ln3) login_node=bigpurple3.nyumc.org ;;
  bigpurple-ln4) login_node=bigpurple4.nyumc.org ;;
esac
if [ -z "$login_node" ]; then
  login_node=bigpurple3.nyumc.org
fi

echo -e "
MacOS or linux terminal command to create your ssh tunnel:
ssh -N -L ${port}:${node}:${port} ${user}@${login_node}

Then open in your browser:
http://localhost:${port}
"

# ---------------------- Environment ----------------------
module load anaconda3
source /gpfs/share/apps/anaconda3/gpu/2023.09/etc/profile.d/conda.sh
# conda deactivate
conda activate /gpfs/data/brandeslab/User/as12267/.conda/envs/huggingface_bert
echo "Conda env: $CONDA_PREFIX"

# Optional GPU info
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "No GPU detected (CPU job)"

# ---------------------- Start Notebook ----------------------
jupyter-notebook --no-browser --port=${port} --ip=${node}
