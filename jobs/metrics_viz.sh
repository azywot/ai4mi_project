#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=segthor_train
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/metrics_viz_%j.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Activate your environment
source ai4mi/bin/activate

python metrics_viz.py \
  --folders val_baseline37 val_baseline42 val_baseline420 \
            val_prep_37 val_prep_42 val_prep_420 \
  --out metrics_compare
