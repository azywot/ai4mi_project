#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:50:00
#SBATCH --output=outfiles/test_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ai4mi/bin/activate
#python main.py --dataset SEGTHOR_CLEAN --mode full --epoch 1 --dest train_results --gpu

# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_42_alisia --gpu --wandb_entity alisia.baielli@student.uva.nl --wandb_project ai4med --seed 42 --wandb_name "baseline_seed_42"
# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_420 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 420 --wandb_name "baseline_seed_420"
# python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest train_results/train_results_baseline_37 --gpu --wandb_entity michal.mazuryk@student.uva.nl --wandb_project ai4med --seed 37 --wandb_name "baseline_seed_37"

# python combined_plot.py --results_dir train_results/train_results_baseline_42 --output plot_baseline_42.pdf


# python stitch.py \
#   --data_folder train_results/train_results_baseline_42/best_epoch/val \
#   --dest_folder val_baseline42/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# mkdir -p val_baseline42/gt

# python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val_baseline42/gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

# python distorch/compute_metrics.py --ref_folder val_baseline42/gt --pred_folder val_baseline42/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val_baseline42






# python combined_plot.py --results_dir train_results/train_results_baseline_420 --output plot_baseline_420.pdf


# python stitch.py \
#   --data_folder train_results/train_results_baseline_420/best_epoch/val \
#   --dest_folder val_baseline420/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# mkdir -p val_baseline420/gt

# python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val_baseline420/gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

# python distorch/compute_metrics.py --ref_folder val_baseline420/gt --pred_folder val_baseline420/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val_baseline420




python combined_plot.py --results_dir train_results/train_results_baseline_37 --output plot_baseline_37.pdf


python stitch.py \
  --data_folder train_results/train_results_baseline_37/best_epoch/val \
  --dest_folder val_baseline37/pred \
  --num_classes 5 \
  --grp_regex "(Patient_\d\d)_\d\d\d\d" \
  --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

mkdir -p val_baseline37/gt

python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder val_baseline37/gt --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

python distorch/compute_metrics.py --ref_folder val_baseline37/gt --pred_folder val_baseline37/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder val_baseline37