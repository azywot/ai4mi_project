#!/usr/bin/env python3
"""
Script to log evaluation metrics to wandb.
Usage: python log_metrics_to_wandb.py --seed SEED --optimizer OPTIMIZER --metrics_dir PATH
"""

import argparse
import os
import wandb
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser(description="Log evaluation metrics to wandb")
    parser.add_argument("--seed", type=int, required=True, help="Seed number")
    parser.add_argument("--optimizer", type=str, required=True, help="Optimizer name")
    parser.add_argument("--metrics_dir", type=str, required=True, help="Path to metrics directory")
    parser.add_argument("--project", type=str, default="final-project", help="Wandb project name")
    parser.add_argument("--entity", type=str, default="ai4med-team", help="Wandb entity name")
    
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    
    # Initialize wandb for this seed
    wandb.init(
        mode="online",
        project=args.project,
        entity=args.entity, 
        name=f'{args.optimizer}_seed_{args.seed}_metrics',
        job_type='evaluation',
        config={'seed': args.seed, 'optimizer': args.optimizer}
    )
    
    # Load and log metrics
    metrics_dir = Path(args.metrics_dir)
    metrics_to_log = {}
    
    print(f"Loading metrics from: {metrics_dir}")
    
    for metric_file in metrics_dir.glob('*.npy'):
        metric_name = metric_file.stem
        data = np.load(metric_file)
        
        # Log mean values per class
        mean_values = data.mean(axis=0)
        for class_idx, value in enumerate(mean_values):
            metrics_to_log[f'{metric_name}_class_{class_idx}'] = float(value)
        
        # Log overall mean
        metrics_to_log[f'{metric_name}_mean'] = float(data.mean())
        
        print(f'Logged {metric_name}: {mean_values}')
    
    wandb.log(metrics_to_log)
    wandb.finish()
    print('Metrics logged to wandb successfully')

if __name__ == "__main__":
    main()
