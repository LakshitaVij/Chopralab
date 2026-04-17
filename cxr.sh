#!/bin/bash
#SBATCH --job-name=cxr2h5
#SBATCH --array=0-2999%100
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/cxr2h5_%A_%a.out
#SBATCH --error=logs/cxr2h5_%A_%a.err

# -----------------------------
# Setup environment
# -----------------------------
source ~/.bashrc
conda activate dicomh5

# -----------------------------
# Go to project directory
# -----------------------------
cd /gpfs/data/chopralab/lv2255/h5_benchmark

mkdir -p logs
mkdir -p /gpfs/data/oermannlab/private_data/thoracic/cxr/h5_cxr_fullrun

# -----------------------------
# Run conversion
# -----------------------------
python finalct.py \
  --input_dir /gpfs/data/oermannlab/private_data/thoracic/cxr/dicom \
  --out_dir /gpfs/data/oermannlab/private_data/thoracic/cxr/h5_cxr_fullrun \
  --chunk_size 64 \
  --compression blosc \
  --index $SLURM_ARRAY_TASK_ID \
  --num_jobs 3000

