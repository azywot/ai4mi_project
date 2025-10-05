#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=outfiles/postprocessing_3_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ../ai4mi_project_michal/ai4mi/bin/activate

# python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val/val_gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch.py \
#   --data_folder train_results/train_results_baseline_42/best_epoch/val \
#   --dest_folder val/val_42/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch.py \
#   --data_folder train_results/train_results_baseline_37/best_epoch/val \
#   --dest_folder val/val_37/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch.py \
#   --data_folder train_results/train_results_baseline_420/best_epoch/val \
#   --dest_folder val/val_420/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# Morphological operations (strongest effect)
python advanced_postprocessing.py \
    --input_folder val/val_37/pred \
    --output_folder val/val_37/pred_morphological \
    --num_classes 5 \
    --method morphological \
    --min_size 100

# Connected component analysis
# python advanced_postprocessing.py \
#     --input_folder val/val_37/pred \
#     --output_folder val/val_37/pred_connected \
#     --num_classes 5 \
#     --method connected

# Multi-resolution processing
python advanced_postprocessing.py \
    --input_folder val/val_37/pred \
    --output_folder val/val_37/pred_multi_res \
    --num_classes 5 \
    --method multi_resolution \
    --min_size 100

python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_37/pred_morphological --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_37/pred_morphological
# python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_37/pred_connected --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_37/pred_connected
python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_37/pred_multi_res --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_37/pred_multi_res
