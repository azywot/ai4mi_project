#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:30:00
#SBATCH --output=outfiles/postprocessing_epoch_5_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ../ai4mi_project_michal/ai4mi/bin/activate

# python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val/val_gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

python stitch.py \
  --data_folder train_results/train_results_baseline_42/iter010/val \
  --dest_folder val/val_42/pred_epoch_10 \
  --num_classes 5 \
  --grp_regex "(Patient_\d\d)_\d\d\d\d" \
  --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

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

python simple_crf.py --input_folder val/val_42/pred_epoch_10 --output_folder val/val_42/pred_crf_epoch_10 --num_classes 5 --spatial_weight 1.0 --num_iterations 10
#python simple_crf.py --input_folder val/val_37/pred --output_folder val/val_37/pred_crf_10 --num_classes 5 --spatial_weight 1.0 --num_iterations 10
#python simple_crf.py --input_folder val/val_420/pred --output_folder val/val_420/pred_crf_10 --num_classes 5 --spatial_weight 1.0 --num_iterations 10

python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_42/pred_epoch_10 --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_42/pred_epoch_10
python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_42/pred_crf_epoch_10 --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_42/crf_epoch_10

#python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_37/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_37
# python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_37/pred_crf_10 --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_37/crf_10

#python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_420/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_420
# python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder val/val_420/pred_crf_10 --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val/val_420/crf_10
