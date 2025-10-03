#!/usr/bin/env python3
"""
Summary script to compare AdamW results across all seeds.
This script reads the metrics computed for each seed and creates a per-class summary.
Each class shows mean ± std computed across all seeds.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

def load_metrics_from_npy(metrics_dir):
    """Load metrics from .npy files in a seed directory."""
    metrics_dir = Path(metrics_dir)
    
    if not metrics_dir.exists():
        print(f"Warning: Directory {metrics_dir} does not exist")
        return {}
    
    metrics = {}
    for npy_file in metrics_dir.glob("*.npy"):
        metric_name = npy_file.stem
        data = np.load(npy_file)
        metrics[metric_name] = data
    
    return metrics

def compute_per_class_stats(all_seed_metrics):
    """
    Compute per-class statistics (mean and std) across all seeds.
    
    For each class, we compute:
    - Mean across all samples within each seed
    - Then mean and std of those means across seeds
    
    Args:
        all_seed_metrics: dict of {seed: {metric_name: data_array}}
    
    Returns:
        DataFrame with columns: class, dice_mean, dice_std, hd95_mean, hd95_std, etc.
    """
    if not all_seed_metrics:
        return None
    
    # Get metric names from first seed
    first_seed = list(all_seed_metrics.keys())[0]
    metric_names = list(all_seed_metrics[first_seed].keys())
    
    # Get number of classes from first metric
    first_metric = list(all_seed_metrics[first_seed].values())[0]
    num_classes = first_metric.shape[1] if len(first_metric.shape) > 1 else first_metric.shape[0]
    
    # Initialize results
    results = []
    
    for class_idx in range(num_classes):
        row = {'class': class_idx}
        
        for metric_name in sorted(metric_names):
            # Collect values for this class across all seeds
            class_values_per_seed = []
            
            for seed, metrics in all_seed_metrics.items():
                if metric_name in metrics:
                    data = metrics[metric_name]
                    # data shape is (num_samples, num_classes)
                    # We want mean across samples for this class, for this seed
                    if len(data.shape) > 1:
                        class_mean = data[:, class_idx].mean()
                    else:
                        class_mean = data[class_idx]
                    class_values_per_seed.append(class_mean)
            
            # Compute mean and std across seeds for this class
            if class_values_per_seed:
                # Convert metric name: 3d_dice -> dice
                short_name = metric_name.replace('3d_', '')
                row[f'{short_name}_mean'] = np.mean(class_values_per_seed)
                row[f'{short_name}_std'] = np.std(class_values_per_seed)
        
        results.append(row)
    
    return pd.DataFrame(results)

def compare_with_baseline(adamw_df, baseline_file):
    """Compare AdamW per-class results with baseline per-class results."""
    if not Path(baseline_file).exists():
        print(f"Warning: Baseline file {baseline_file} not found")
        return None
    
    baseline_df = pd.read_csv(baseline_file)
    
    # Create comparison DataFrame
    comparison = []
    
    for class_idx in adamw_df['class']:
        adamw_row = adamw_df[adamw_df['class'] == class_idx].iloc[0]
        baseline_row = baseline_df[baseline_df['class'] == class_idx].iloc[0]
        
        comp_row = {'class': class_idx}
        
        # Compare each metric
        for col in adamw_df.columns:
            if col != 'class' and '_mean' in col:
                metric_name = col.replace('_mean', '')
                adamw_mean = adamw_row[col]
                baseline_mean = baseline_row[col]
                
                # Calculate improvement
                improvement = adamw_mean - baseline_mean
                improvement_pct = (improvement / baseline_mean * 100) if baseline_mean != 0 else 0
                
                comp_row[f'{metric_name}_adamw'] = adamw_mean
                comp_row[f'{metric_name}_baseline'] = baseline_mean
                comp_row[f'{metric_name}_diff'] = improvement
                comp_row[f'{metric_name}_diff_pct'] = improvement_pct
        
        comparison.append(comp_row)
    
    return pd.DataFrame(comparison)

def main():
    parser = argparse.ArgumentParser(description="Summarize AdamW results across seeds")
    parser.add_argument("--results_dir", 
                       default="train_results/adamW", 
                       help="Directory containing training results (train_results/adamW)")
    parser.add_argument("--output_file", 
                       default="train_results/adamW/per_class_stats.csv", 
                       help="Output CSV file for per-class summary")
    parser.add_argument("--comparison_file",
                       default="train_results/adamW/comparison_with_baseline.csv",
                       help="Output CSV file for comparison with baseline")
    parser.add_argument("--baseline_file",
                       default="train_results/per_class_stats.csv",
                       help="Baseline per-class stats file")
    parser.add_argument("--seeds", 
                       nargs="+", 
                       default=[42, 420, 37], 
                       type=int,
                       help="Seeds to include in summary")
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    seeds = args.seeds
    
    print(f"Summarizing results for seeds: {seeds}")
    print(f"Results directory: {results_dir}")
    
    # Collect metrics for each seed
    all_seed_metrics = {}
    
    for seed in seeds:
        # Look for metrics in train_results/adamW/train_results_baseline_{seed}/...
        seed_dir = results_dir / f"train_results_baseline_{seed}"
        
        # Try evaluation_results location first
        eval_metrics_dir = Path("evaluation_results/adamW/metrics") / f"seed_{seed}"
        
        metrics_dir = None
        if eval_metrics_dir.exists():
            metrics_dir = eval_metrics_dir
        elif seed_dir.exists():
            # Look for metrics in subdirectories
            possible_paths = [
                seed_dir / "metrics",
                seed_dir / "best_epoch" / "metrics",
            ]
            for path in possible_paths:
                if path.exists():
                    metrics_dir = path
                    break
        
        if metrics_dir is None:
            print(f"\nWarning: Could not find metrics directory for seed {seed}")
            print(f"  Tried: {eval_metrics_dir}")
            print(f"  Tried: {seed_dir}")
            continue
        
        print(f"\nProcessing seed {seed} from {metrics_dir}")
        
        metrics = load_metrics_from_npy(metrics_dir)
        if metrics:
            all_seed_metrics[seed] = metrics
            print(f"Loaded {len(metrics)} metrics for seed {seed}")
            for metric_name, data in metrics.items():
                print(f"  {metric_name}: shape {data.shape}")
        else:
            print(f"No metrics found for seed {seed}")
    
    if not all_seed_metrics:
        print("\nNo results found for any seed!")
        return
    
    # Compute per-class statistics (averaged over seeds)
    print("\n" + "="*60)
    print("Computing per-class statistics (mean ± std over seeds)...")
    per_class_df = compute_per_class_stats(all_seed_metrics)
    
    if per_class_df is None:
        print("Failed to compute per-class statistics")
        return
    
    # Save to CSV
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    per_class_df.to_csv(output_file, index=False)
    
    print(f"\nPer-class statistics saved to: {output_file}")
    print("\nAdamW Per-Class Results (mean ± std over seeds):")
    print("="*60)
    print(per_class_df.to_string(index=False, float_format='%.6f'))
    
    # Compare with baseline
    print("\n" + "="*60)
    print("Comparing with baseline (per-class)...")
    comparison_df = compare_with_baseline(per_class_df, args.baseline_file)
    
    if comparison_df is not None:
        comparison_file = Path(args.comparison_file)
        comparison_file.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(comparison_file, index=False)
        
        print(f"\nComparison saved to: {comparison_file}")
        print("\nPer-Class Comparison with Baseline:")
        print("="*60)
        print(comparison_df.to_string(index=False, float_format='%.4f'))

if __name__ == "__main__":
    main()
