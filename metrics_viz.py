#!/usr/bin/env python3
"""
Compare segmentation metrics across Baseline vs Preprocessed runs.

Usage:
    python metrics_viz.py \
        --folders val_baseline37 val_baseline42 val_baseline420 \
                  val_prep37 val_prep42 val_prep420 \
        --out metrics_compare

Generates:
    - One radar plot per metric comparing baseline vs prep
    - One violin plot per metric per group
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# ------------------------------------
# Class labels and color schemes
# ------------------------------------
class_names = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]
colors = {
    "baseline": "#ff7f0e",  # orange
    "prep": "#1f77b4",      # blue
}
metrics = ["3d_dice", "3d_hd95", "3d_jaccard", "3d_assd"]

# ------------------------------------
# Helpers
# ------------------------------------
def load_group_data(group_folders, metric):
    """Load and average metrics across seeds for one group (baseline/prep)."""
    all_data = []
    for f in group_folders:
        fpath = Path(f) / f"{metric}.npy"
        if fpath.exists():
            data = np.load(fpath)  # (n_patients, n_classes)
            all_data.append(data)
            print(f"Loaded {metric} from {f} with shape {data.shape}")
        else:
            print(f"Warning: {fpath} not found.")
    if not all_data:
        return None
    stacked = np.stack(all_data, axis=0)  # (n_seeds, n_patients, n_classes)
    return stacked.mean(axis=0)  # average across seeds


def save_violin(data, title, ylabel, filename, color, out_folder):
    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot([data[:, i] for i in range(data.shape[1])],
                          showmeans=True, showmedians=True)

    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)
    parts['cmeans'].set_color("black")
    parts['cmedians'].set_color("red")

    ax.set_xticks(range(1, data.shape[1] + 1))
    ax.set_xticklabels(class_names, rotation=25)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_folder / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")


def save_radar(baseline_data, prep_data, title, filename, out_folder):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    N = len(class_names)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist() + [0]

    for group, data in [("baseline", baseline_data), ("prep", prep_data)]:
        if data is None:
            continue
        means = np.nanmean(data, axis=0).tolist() + [np.nanmean(data, axis=0)[0]]
        ax.plot(angles, means, 'o-', linewidth=2, label=group.capitalize(),
                color=colors[group])
        ax.fill(angles, means, color=colors[group], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_title(title, size=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig(out_folder / filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")

# ------------------------------------
# Main
# ------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Visualize metrics comparing baseline vs preprocessing.")
    parser.add_argument("--folders", type=str, nargs="+", required=True,
                        help="Folders containing metric .npy files (both baseline and prep).")
    parser.add_argument("--out", type=str, default="metrics_compare",
                        help="Output folder for plots.")
    args = parser.parse_args()

    out_folder = Path(args.out)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Group folders by prefix
    baseline_folders = [f for f in args.folders if "baseline" in f.lower()]
    prep_folders = [f for f in args.folders if "prep" in f.lower()]

    print(f"Found {len(baseline_folders)} baseline folders and {len(prep_folders)} prep folders.")

    # Load and average each group
    metric_data = {}
    for m in metrics:
        metric_data[m] = {
            "baseline": load_group_data(baseline_folders, m),
            "prep": load_group_data(prep_folders, m)
        }

    # --- Generate plots ---
    for m in metrics:
        print(f"\n=== {m.upper()} ===")
        base = metric_data[m]["baseline"]
        prep = metric_data[m]["prep"]

        if base is not None:
            save_violin(base, f"{m} per class (Baseline)", m, f"{m}_violin_baseline.png",
                        color=colors["baseline"], out_folder=out_folder)
        if prep is not None:
            save_violin(prep, f"{m} per class (Preprocessed)", m, f"{m}_violin_prep.png",
                        color=colors["prep"], out_folder=out_folder)

        if base is not None and prep is not None:
            save_radar(base, prep, f"{m} mean per class", f"{m}_radar_compare.png", out_folder)

    print(f"\nAll plots saved in: {out_folder}")

if __name__ == "__main__":
    main()
