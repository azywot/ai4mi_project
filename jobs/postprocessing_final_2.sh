#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=1
#SBATCH --job-name=Eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=01:30:00
#SBATCH --output=outfiles/postprocessing_final_2_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2
module load CUDA/12.4.0
#

source ai4mi/bin/activate


# Set your input and output directories
# INPUT_FOLDER=("val/val_37/pred" "val/val_42/pred" "val/val_420/pred")
# BASE_OUTPUT=("val/val_37" "val/val_42" "val/val_420")

INPUT_FOLDER=("/home/abaielli/ai4mi_project/best/pred_37" "/home/abaielli/ai4mi_project/best/pred_42" "/home/abaielli/ai4mi_project/best/pred_420")
BASE_OUTPUT=("/home/abaielli/ai4mi_project/postprocessing_results_final/37" "/home/abaielli/ai4mi_project/postprocessing_results_final/42" "/home/abaielli/ai4mi_project/postprocessing_results_final/420")

for i in ${!INPUT_FOLDER[@]}; do
    INPUT_FOLDER_I=${INPUT_FOLDER[$i]}
    BASE_OUTPUT_I=${BASE_OUTPUT[$i]}
    echo "Input folder: ${INPUT_FOLDER_I}"
    echo "Base output: ${BASE_OUTPUT_I}"
    echo ""

    # Final Method: Optimized pipeline (MCA on 0,1,2,4 -> HD optimized -> MCA on 3)
    echo "Final Method: MCA (classes 0,1,2,4) -> HD optimized -> MCA (class 3 only)"
    python postprocessing.py --input_folder $INPUT_FOLDER_I --output_folder $BASE_OUTPUT_I/mca --method mca --num_classes 5 --min_component_size 50 --artifact_threshold 0.6 --connectivity_threshold 0.3
    #echo "Computing metrics for Final method..."
    #python distorch/compute_metrics.py --ref_folder val/gt --pred_folder $BASE_OUTPUT_I/mca --ref_extension .nii.gz --pred_extension .nii.gz --num_classes 5 --metrics 3d_dice 3d_hd95 3d_jaccard 3d_assd --save_folder $BASE_OUTPUT_I/mca --overwrite
    #echo ""


    echo "=========================================="
    echo "FINAL OPTIMIZED METHOD:"
    echo "=========================================="
    echo "FINAL METHOD:"
    echo "Final: MCA (classes 0,1,2,4) -> HD optimized -> MCA (class 3 only): $BASE_OUTPUT_I/final"
    echo ""
    echo "PIPELINE STEPS:"
    echo "1. MCA on classes 0,1,2,4 (skip trachea class 3)"
    echo "2. HD optimized processing on all classes"
    echo "3. MCA on class 3 (trachea) only"
    echo ""
    echo "EXPECTED BENEFITS:"
    echo "- Better performance on esophagus (class 1) with minimal processing"
    echo "- Improved HD metrics through aggressive smoothing"
    echo "- Optimized trachea (class 3) processing as final step"
    echo "=========================================="
done

