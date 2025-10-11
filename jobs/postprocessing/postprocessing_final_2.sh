#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=00:20:00
#SBATCH --output=outfiles/postprocessing_final_2_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ../ai4mi_project_michal/ai4mi/bin/activate

echo "=== Ensemble Post-Processing for Medical Images ==="
echo ""

# Set your input and output directories
# INPUT_FOLDER=("val/val_37/pred" "val/val_42/pred" "val/val_420/pred")
# BASE_OUTPUT=("val/val_37" "val/val_42" "val/val_420")

INPUT_FOLDER=("val_4/37/pred")
BASE_OUTPUT=("val_4/37")

for i in ${!INPUT_FOLDER[@]}; do
    INPUT_FOLDER_I=${INPUT_FOLDER[$i]}
    BASE_OUTPUT_I=${BASE_OUTPUT[$i]}
    echo "Input folder: ${INPUT_FOLDER_I}"
    echo "Base output: ${BASE_OUTPUT_I}"
    echo ""

    # Method 5: Combined Processing (All Methods)
    # echo "5. Running Combined Ensemble Post-Processing..."
    # python ensemble_postprocessing.py \
    #     --input_folder "$INPUT_FOLDER_I" \
    #     --output_folder "${BASE_OUTPUT_I}/pred_combined_ensemble" \
    #     --num_classes 5 \
    #     --method combined \
    #     --roi_size 128

    # echo ""

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

    echo "Computing metrics for Baseline..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred --overwrite

    echo "Computing metrics for Combined Ensemble Processing..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_combined_ensemble --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_combined_ensemble --overwrite

    echo "Computing metrics for Simple CRF Post-Processing..."
    python distorch/compute_metrics.py --ref_folder val/val_gt --pred_folder $folder/pred_simple_crf --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $folder/pred_simple_crf --overwrite

    echo "----------------------------------------"
done