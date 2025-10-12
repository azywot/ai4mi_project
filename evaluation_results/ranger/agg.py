#!/usr/bin/env python3
"""
Script to aggregate results from all Ranger loss types and create a summary CSV
with averages and percentage differences.
"""

import pandas as pd
import numpy as np
import os
import glob

def aggregate_ranger_results():
    """Aggregate results from all Ranger loss types."""
    
    # Directory containing the CSV files
    results_dir = "/scratch-shared/ai4mi-group-9/ai4mi_project/evaluation_results/ranger"
    
    # Find all comparison CSV files
    csv_files = glob.glob(os.path.join(results_dir, "cedice_comparison_with_baseline.csv"))
    
    print(f"Found {len(csv_files)} CSV files to process:")
    for file in csv_files:
        print(f"  - {os.path.basename(file)}")
    
    # Initialize lists to store dataframes
    all_dataframes = []
    
    # Read each CSV file
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            # Add a column to identify the loss type
            loss_type = os.path.basename(csv_file).replace('_comparison_with_baseline.csv', '')
            df['loss_type'] = loss_type
            all_dataframes.append(df)
            print(f"Successfully loaded {loss_type}: {len(df)} rows")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if not all_dataframes:
        print("No valid CSV files found!")
        return
    
    # Combine all dataframes
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    print(f"\nCombined dataset: {len(combined_df)} total rows")
    
    # Calculate averages for each class across all loss types
    metrics = ['assd_optimizer', 'assd_baseline', 'dice_optimizer', 'dice_baseline', 
               'hd95_optimizer', 'hd95_baseline', 'jaccard_optimizer', 'jaccard_baseline']
    
    # Group by class and calculate means
    class_averages = combined_df.groupby('class')[metrics].mean().reset_index()
    
    # Calculate differences and percentage differences
    class_averages['assd_diff'] = class_averages['assd_optimizer'] - class_averages['assd_baseline']
    class_averages['assd_diff_pct'] = (class_averages['assd_diff'] / class_averages['assd_baseline']) * 100
    
    class_averages['dice_diff'] = class_averages['dice_optimizer'] - class_averages['dice_baseline']
    class_averages['dice_diff_pct'] = (class_averages['dice_diff'] / class_averages['dice_baseline']) * 100
    
    class_averages['hd95_diff'] = class_averages['hd95_optimizer'] - class_averages['hd95_baseline']
    class_averages['hd95_diff_pct'] = (class_averages['hd95_diff'] / class_averages['hd95_baseline']) * 100
    
    class_averages['jaccard_diff'] = class_averages['jaccard_optimizer'] - class_averages['jaccard_baseline']
    class_averages['jaccard_diff_pct'] = (class_averages['jaccard_diff'] / class_averages['jaccard_baseline']) * 100
    
    # Add a summary row with overall averages
    summary_row = {
        'class': 'Overall Average',
        'assd_optimizer': class_averages['assd_optimizer'].mean(),
        'assd_baseline': class_averages['assd_baseline'].mean(),
        'assd_diff': class_averages['assd_diff'].mean(),
        'assd_diff_pct': (class_averages['assd_diff'].mean() / class_averages['assd_baseline'].mean()) * 100,
        'dice_optimizer': class_averages['dice_optimizer'].mean(),
        'dice_baseline': class_averages['dice_baseline'].mean(),
        'dice_diff': class_averages['dice_diff'].mean(),
        'dice_diff_pct': (class_averages['dice_diff'].mean() / class_averages['dice_baseline'].mean()) * 100,
        'hd95_optimizer': class_averages['hd95_optimizer'].mean(),
        'hd95_baseline': class_averages['hd95_baseline'].mean(),
        'hd95_diff': class_averages['hd95_diff'].mean(),
        'hd95_diff_pct': (class_averages['hd95_diff'].mean() / class_averages['hd95_baseline'].mean()) * 100,
        'jaccard_optimizer': class_averages['jaccard_optimizer'].mean(),
        'jaccard_baseline': class_averages['jaccard_baseline'].mean(),
        'jaccard_diff': class_averages['jaccard_diff'].mean(),
        'jaccard_diff_pct': (class_averages['jaccard_diff'].mean() / class_averages['jaccard_baseline'].mean()) * 100
    }
    
    # Add the summary row
    summary_df = pd.DataFrame([summary_row])
    final_df = pd.concat([class_averages, summary_df], ignore_index=True)
    
    # Format percentage columns to show as percentages
    pct_columns = ['assd_diff_pct', 'dice_diff_pct', 'hd95_diff_pct', 'jaccard_diff_pct']
    for col in pct_columns:
        final_df[col] = final_df[col].round(2).astype(str) + '%'
    
    # Save the aggregated results
    output_file = os.path.join(results_dir, "ranger_aggregated_comparison_with_baseline.csv")
    final_df.to_csv(output_file, index=False)
    
    print(f"\nAggregated results saved to: {output_file}")
    print(f"Summary:")
    print(f"  - Processed {len(csv_files)} loss types")
    print(f"  - {len(class_averages)} classes analyzed")
    print(f"  - Overall averages calculated")
    
    # Display the summary row
    print(f"\nOverall Averages:")
    print(f"  ASSD: {summary_row['assd_optimizer']:.4f} vs {summary_row['assd_baseline']:.4f} ({summary_row['assd_diff_pct']:.2f}%)")
    print(f"  Dice: {summary_row['dice_optimizer']:.4f} vs {summary_row['dice_baseline']:.4f} ({summary_row['dice_diff_pct']:.2f}%)")
    print(f"  HD95: {summary_row['hd95_optimizer']:.4f} vs {summary_row['hd95_baseline']:.4f} ({summary_row['hd95_diff_pct']:.2f}%)")
    print(f"  Jaccard: {summary_row['jaccard_optimizer']:.4f} vs {summary_row['jaccard_baseline']:.4f} ({summary_row['jaccard_diff_pct']:.2f}%)")

if __name__ == "__main__":
    aggregate_ranger_results()