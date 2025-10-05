#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=03:30:00
#SBATCH --output=outfiles/postprocessing_final_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ../ai4mi_project_michal/ai4mi/bin/activate

# echo "=== Stiching and CRF Post-Processing for Medical Images ==="
# echo ""

# python stitch_new.py \
#     --data_folder data/SEGTHOR_CLEAN/val/gt \
#     --dest_folder val/val_gt \
#     --num_classes 5 \
#     --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#     --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch_new.py \
#   --data_folder train_results/train_results_baseline_42/best_epoch/val \
#   --dest_folder val/val_42/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch_new.py \
#   --data_folder train_results/train_results_baseline_37/best_epoch/val \
#   --dest_folder val/val_37/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

# python stitch_new.py \
#   --data_folder train_results/train_results_baseline_420/best_epoch/val \
#   --dest_folder val/val_420/pred \
#   --num_classes 5 \
#   --grp_regex "(Patient_\d\d)_\d\d\d\d" \
#   --source_scan_pattern "data/segthor_train/train/{id_}/GT.nii.gz"

echo "=== Ensemble Post-Processing for Medical Images ==="
echo ""

# Set your input and output directories
# INPUT_FOLDER=("val/val_37/pred" "val/val_42/pred" "val/val_420/pred")
# BASE_OUTPUT=("val/val_37" "val/val_42" "val/val_420")

INPUT_FOLDER=("val/val_42/pred" "val/val_420/pred")
BASE_OUTPUT=("val/val_42" "val/val_420")

for i in ${!INPUT_FOLDER[@]}; do
    INPUT_FOLDER_I=${INPUT_FOLDER[$i]}
    BASE_OUTPUT_I=${BASE_OUTPUT[$i]}
    echo "Input folder: ${INPUT_FOLDER_I}"
    echo "Base output: ${BASE_OUTPUT_I}"
    echo ""

    

    # Method 1: Ensemble Voting (from mutil_label_merge.ipynb)
    echo "1. Running Ensemble Voting Post-Processing..."
    python ensemble_postprocessing.py \
        --input_folder "$INPUT_FOLDER_I" \
        --output_folder "${BASE_OUTPUT_I}/pred_ensemble" \
        --num_classes 5 \
        --method ensemble

    echo ""

    # Method 2: ROI-based Processing (from pred_sample.ipynb)
    echo "2. Running ROI-based Post-Processing..."
    python ensemble_postprocessing.py \
        --input_folder "$INPUT_FOLDER_I" \
        --output_folder "${BASE_OUTPUT_I}/pred_roi" \
        --num_classes 5 \
        --method roi \
        --roi_size 128

    echo ""

    # Method 3: Center Constraints (from get_center_2d.ipynb, get_center_3d.ipynb)
    echo "3. Running Center Constraint Post-Processing..."
    python ensemble_postprocessing.py \
        --input_folder "$INPUT_FOLDER_I" \
        --output_folder "${BASE_OUTPUT_I}/pred_center" \
        --num_classes 5 \
        --method center

    echo ""

    # Method 4: Multi-label Fusion (from mutil_label_merge.ipynb)
    echo "4. Running Multi-label Fusion Post-Processing..."
    python ensemble_postprocessing.py \
        --input_folder "$INPUT_FOLDER_I" \
        --output_folder "${BASE_OUTPUT_I}/pred_fusion" \
        --num_classes 5 \
        --method fusion

    echo ""

    # Method 5: Combined Processing (All Methods)
    echo "5. Running Combined Ensemble Post-Processing..."
    python ensemble_postprocessing.py \
        --input_folder "$INPUT_FOLDER_I" \
        --output_folder "${BASE_OUTPUT_I}/pred_combined_ensemble" \
        --num_classes 5 \
        --method combined \
        --roi_size 128

    echo ""

    echo "Applying Simple CRF Post-Processing..."

    python simple_crf.py \
    --input_folder "$INPUT_FOLDER_I" \
    --output_folder "${BASE_OUTPUT_I}/pred_simple_crf" \
    --num_classes 5 \
    --spatial_weight 1.0 \
    --num_iterations 5

    echo "----------------------------------------"
done

echo "Computing metrics..."

for i in ${!BASE_OUTPUT[@]}; do
    folder=${BASE_OUTPUT[$i]}
    echo "Computing metrics for folder: $folder"

    echo "Computing metrics for Ensemble Voting..." 
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_ensemble --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd \
    --save_folder $folder/pred_ensemble

    echo "Computing metrics for ROI-based Processing..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_roi --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_roi

    echo "Computing metrics for Center Constraints..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_center --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_center

    echo "Computing metrics for Multi-label Fusion..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_fusion --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_fusion

    echo "Computing metrics for Combined Ensemble Processing..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_combined_ensemble --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_combined_ensemble

    echo "Computing metrics for Simple CRF Post-Processing..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_simple_crf --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_simple_crf

    echo "----------------------------------------"
done