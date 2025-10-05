#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=02:00:00
#SBATCH --output=outfiles/full_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

cd $HOME/group_project/ai4mi_project/
source ai4mi/bin/activate

make data/SEGTHOR_CLEAN CFLAGS=-O -n  # Will display the commands that will run, easy to inspect:
rm -rf data/segthor_fixed_tmp data/segthor_fixed
python -O sabotage.py --mode inv --source_dir data/segthor_train --dest_dir data/segthor_fixed_tmp -K 2 --regex_gt "GT.nii.gz" -p 4
mv data/segthor_fixed_tmp data/segthor_fixed
rm -rf data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN
python -O slice_segthor.py --source_dir data/segthor_fixed --dest_dir data/SEGTHOR_CLEAN_tmp \
        --shape 256 256 --retain 10 -p -1
mv data/SEGTHOR_CLEAN_tmp data/SEGTHOR_CLEAN


# Define seeds array
SEEDS=(42 420 37)

# Train models for all seeds
for seed in "${SEEDS[@]}"; do
    echo "Training with seed $seed"
    python -O main.py --dataset SEGTHOR_CLEAN --mode full --epoch 25 --dest "train_results_baseline_${seed}" --gpu --wandb_entity ai4med-team --wandb_project final-project --seed $seed --wandb_name "baseline_seed_${seed}"
done

# Plot results for all seeds
for seed in "${SEEDS[@]}"; do
    echo "Plotting results for seed $seed"
    python plot_full.py --result_folder "train_results_baseline_${seed}" --output_pdf "plot_full_seed_${seed}.pdf"
done

# Stitch and compute metrics for all seeds
for seed in "${SEEDS[@]}"; do
    echo "Processing results for seed $seed"

    # Create seed-specific validation directories
    mkdir -p "val_seed_${seed}/pred"
    mkdir -p "val_seed_${seed}/gt"

    # Stitch predictions for this seed
    python stitch.py \
      --data_folder "train_results_baseline_${seed}/best_epoch/val" \
      --dest_folder "val_seed_${seed}/pred" \
      --num_classes 5 \
      --grp_regex "(Patient_\d\d)_\d\d\d\d" \
      --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

    # Stitch ground truth for this seed
    python stitch.py --data_folder data/SEGTHOR_CLEAN/val/gt --dest_folder "val_seed_${seed}/gt" --num_classes 5 --grp_regex "(Patient_\d\d)_\d\d\d\d" --source_scan_pattern "data/segthor_fixed/train/{id_}/GT.nii.gz"

    # Compute metrics for this seed
    python distorch/compute_metrics.py --ref_folder "val_seed_${seed}/gt" --pred_folder "val_seed_${seed}/pred" --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 --save_folder "val_seed_${seed}"
done