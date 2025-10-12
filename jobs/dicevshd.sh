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


python dicevshd.py \
  --group "Baseline:/home/scur1014/ai4mi_project/train_results_baseline/baseline_42/metrics,/home/scur1014/ai4mi_project/train_results_baseline/baseline_37/metrics,/home/scur1014/ai4mi_project/train_results_baseline/baseline_420/metrics" \
  --group "Full:/home/scur1014/ai4mi_project/prep_transU_ranger_cedice_sched_gradclip/prep_transU_ranger_cedice_sched_gradclip_42/metrics,/home/scur1014/ai4mi_project/prep_transU_ranger_cedice_sched_gradclip/prep_transU_ranger_cedice_sched_gradclip_37/metrics,/home/scur1014/ai4mi_project/prep_transU_ranger_cedice_sched_gradclip/prep_transU_ranger_cedice_sched_gradclip_420/metrics" \
  --group "Ablation (no preprocessing):/home/scur1014/ai4mi_project/transU-ranger-cedice-sched-grad-clip_results/transU-ranger-cedice-sched-grad-clip_42/metrics,/home/scur1014/ai4mi_project/transU-ranger-cedice-sched-grad-clip_results/transU-ranger-cedice-sched-grad-clip_37/metrics,/home/scur1014/ai4mi_project/transU-ranger-cedice-sched-grad-clip_results/transU-ranger-cedice-sched-grad-clip_420/metrics" \
  --out plots/dsc_hd95_baseline_vs_full