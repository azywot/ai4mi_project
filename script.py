import numpy as np
import re
from collections import defaultdict

def parse_metrics(file_path):
    metrics = defaultdict(list)

    # Regex to capture metric name and values inside brackets
    pattern = re.compile(r'(\w+)\s*\[(.*?)\]')

    with open(file_path, 'r') as f:
        for line in f:
            match = pattern.search(line.strip())
            if match:
                name, values = match.groups()
                arr = np.fromstring(values.replace('\n', ''), sep=' ')
                metrics[name].append(arr)
    
    return metrics

def compute_mean_std(metrics):
    results = {}
    for metric, arrays in metrics.items():
        stacked = np.vstack(arrays)  # shape (num_seeds, num_classes)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        results[metric] = (mean, std)
    return results

def print_results(results):
    for metric, (mean, std) in results.items():
        print(f"\nMetric: {metric}")
        for i, (m, s) in enumerate(zip(mean, std), 1):
            print(f"  Class {i}: mean={m:.4f}, std={s:.4f}")

if __name__ == "__main__":
    file_path = "results.txt"  # <-- put your txt file name here
    metrics = parse_metrics(file_path)
    results = compute_mean_std(metrics)
    print_results(results)