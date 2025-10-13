#!/bin/bash
#SBATCH --partition=gpu_h100
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
  --folders /home/abaielli/ai4mi_project/final_baseline/final_baseline_42/metrics /home/abaielli/ai4mi_project/final_baseline/final_baseline_37/metrics /home/abaielli/ai4mi_project/final_baseline/final_baseline_420/metrics \
            /home/abaielli/ai4mi_project/postprocessing_results_final/37/final /home/abaielli/ai4mi_project/postprocessing_results_final/420/final /home/abaielli/ai4mi_project/postprocessing_results_final/42/final \
  --out metrics_compare
