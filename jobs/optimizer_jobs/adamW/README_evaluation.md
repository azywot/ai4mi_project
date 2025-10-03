# AdamW Optimizer Evaluation

This directory contains scripts for programmatically evaluating all seeds for the AdamW optimizer experiment.

## Directory Structure

```
evaluation_results/adamW/
├── ground_truth/                    # Ground truth (same for all seeds)
├── stitched_predictions/
│   ├── seed_42/                    # Stitched predictions for seed 42
│   ├── seed_420/                   # Stitched predictions for seed 420
│   └── seed_37/                    # Stitched predictions for seed 37
├── metrics/
│   ├── seed_42/                    # Metrics for seed 42
│   ├── seed_420/                   # Metrics for seed 420
│   └── seed_37/                    # Metrics for seed 37
└── summary_results.csv             # Summary comparison across seeds
```

## Scripts

### 1. `evaluate_adamw_all_seeds.sh`
- Basic evaluation script that stitches predictions and computes metrics for all seeds
- Uses the same ground truth for all evaluations
- Creates organized directory structure

### 2. `evaluate_and_summarize_adamw.sh` (Recommended)
- Comprehensive evaluation script with error handling and progress reporting
- Includes summary generation
- Provides detailed status updates during execution

### 3. `summarize_adamw_results.py`
- Python script to generate summary statistics across seeds
- Creates CSV file with comparative metrics
- Computes cross-seed statistics (mean, std, min, max)

## Usage

### Run Complete Evaluation (Recommended)
```bash
sbatch evaluate_and_summarize_adamw.sh
```

### Run Basic Evaluation Only
```bash
sbatch evaluate_adamw_all_seeds.sh
```

### Generate Summary Only (if evaluation already completed)
```bash
python summarize_adamw_results.py --results_dir evaluation_results/adamW
```

## Expected Results

The evaluation will:
1. **Prepare ground truth** once (shared across all seeds)
2. **Stitch predictions** for each seed (42, 420, 37)
3. **Compute metrics** (3D Dice, 3D HD95) for each seed
4. **Generate summary** comparing results across seeds

## Output Files

- `summary_results.csv`: Contains mean, std, min, max for each metric across seeds
- Individual metric JSON files in each seed's metrics directory
- Stitched prediction files in NIfTI format

## Prerequisites

- AdamW training results must exist in `train_results/adamW/` directory
- Each seed should have completed training with structure:
  ```
  train_results/adamW/train_results_baseline_{seed}/best_epoch/val/
  ```

## Notes

- All evaluations use the same ground truth for fair comparison
- Scripts include error handling for missing directories/files
- Progress is reported for each step
- Results are organized by seed for easy comparison
