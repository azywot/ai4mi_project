#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=segthor_train
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=logs/preprocessing_%j.out

module purge
module load 2023
module load Anaconda3/2023.07-2

# Activate your environment
source ai4mi/bin/activate

# python -O slice_segthor.py --source_dir data/segthor_fixed --dest_dir data/SEGTHOR_PREP_tmp \
#         --shape 256 256 --retain 10 -p -1 --use_preprocessing
# mv data/SEGTHOR_PREP_tmp data/SEGTHOR_CLEAN


# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results_preprocessing_42 --gpu --wandb_entity alisia.baielli@student.uva.nl --wandb_project ai4med --seed 42 --wandb_name "baseline_seed_42"


python combined_plot.py --results_dir train_results_preprocessing_42 --output plot_prep_42.pdf

# python stitch.py \
#   --data_folder train_results_preprocessing_42/best_epoch/val \
#   --dest_folder val_prep/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# mkdir -p val_prep/gt

# python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val_prep/gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

# python distorch/compute_metrics.py --ref_folder val_prep/gt --pred_folder val_prep/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val_prep