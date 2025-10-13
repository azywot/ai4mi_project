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


python dicevshd.py \
  --group "Baseline:/home/abaielli/ai4mi_project/final_baseline/final_baseline_42/metrics,/home/abaielli/ai4mi_project/final_baseline/final_baseline_37/metrics,/home/abaielli/ai4mi_project/final_baseline/final_baseline_420/metrics" \
  --group "Best Model:/home/abaielli/ai4mi_project/no_preprocessing_full/no_preprocessing_full_42/metrics,/home/abaielli/ai4mi_project/no_preprocessing_full/no_preprocessing_full_37/metrics,/home/abaielli/ai4mi_project/no_preprocessing_full/no_preprocessing_full_420/metrics" \
  --group "Pre-processing + Best Model:/home/abaielli/ai4mi_project/best_model/best_model_37/metrics,/home/abaielli/ai4mi_project/best_model/best_model_420/metrics,/home/abaielli/ai4mi_project/best_model/best_model_420/metrics" \
  --group "Pre_processing + Best Model + Post-processing: /home/abaielli/ai4mi_project/postprocessing_results_final/37/final,/home/abaielli/ai4mi_project/postprocessing_results_final/420/final,/home/abaielli/ai4mi_project/postprocessing_results_final/42/final" \
  --out plots/dsc_hd95_baseline_vs_full




  
#--group "Baseline + Pre-processing: /home/abaielli/ai4mi_project/best_prep/best_prep_42/metrics,/home/abaielli/ai4mi_project/best_prep/best_prep_37/metrics,/home/abaielli/ai4mi_project/best_prep/best_prep_420/metrics" \


  # add final_baseline/final_baseline_42/metrics