#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Class labels
CLASS_NAMES = ["Background", "Esophagus", "Heart", "Trachea", "Aorta"]

# Colors
COLORS = {
    "baseline": "#ff7f0e",   # orange
    "post": "#2ca02c",       # green
}

# Metrics to visualize
METRICS = ["3d_dice", "3d_hd95", "3d_jaccard", "3d_assd"]


def _find_metric_file(base: Path, metric: str) -> Optional[Path]:
    candidates = [
        base / f"{metric}.npy",
        base / f"{metric}.npz",
        base / "metrics" / f"{metric}.npy",
        base / "metrics" / f"{metric}.npz",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _load_metric_array(path: Path) -> Optional[np.ndarray]:
    try:
        if path.suffix == ".npy":
            arr = np.load(path, allow_pickle=True)
        else:
            z = np.load(path, allow_pickle=True)
            # Prefer common keys; otherwise first ndarray
            for k in ["data", "arr_0", "values", "metric", "A"]:
                if k in z.files and isinstance(z[k], np.ndarray):
                    arr = z[k]
                    break
            else:
                for k in z.files:
                    if isinstance(z[k], np.ndarray):
                        arr = z[k]
                        break
                else:
                    return None
        arr = np.asarray(arr)
        # Squeeze and ensure 2D (patients, classes)
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim > 2:
            arr = np.squeeze(arr)
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim != 2:
                return None
        return arr.astype(float)
    except Exception:
        return None


def _ensure_percent_if_prob(arr: np.ndarray) -> np.ndarray:
    finite = np.isfinite(arr)
    if finite.any():
        m, M = float(np.nanmin(arr[finite])), float(np.nanmax(arr[finite]))
        if 0.0 <= m and M <= 1.5:
            return 100.0 * arr
    return arr


def load_group_data(group_folders: List[str], metric: str, require_same_classes: bool = True) -> Optional[np.ndarray]:
    """
    Load and average (across seeds) metrics for a group.
    Returns (P, C) or None.
    Truncates to the smallest number of patients across seeds to align.
    """
    arrays: List[np.ndarray] = []
    for f in group_folders:
        base = Path(f)
        fpath = _find_metric_file(base, metric)
        if fpath is None:
            print(f"[WARN] {metric} not found in {base} (nor metrics/).")
            continue
        arr = _load_metric_array(fpath)
        if arr is None:
            print(f"[WARN] Could not load {metric} from {fpath}.")
            continue
        arrays.append(arr)

    if not arrays:
        return None

    # Align by min patients
    P_min = min(a.shape[0] for a in arrays)
    arrays = [a[:P_min] for a in arrays]

    # Align by classes
    C_set = {a.shape[1] for a in arrays}
    if len(C_set) > 1:
        if require_same_classes:
            print(f"[WARN] Inconsistent class counts for {metric}: {C_set}. Using min classes.")
        C_min = min(C_set)
        arrays = [a[:, :C_min] for a in arrays]

    A = np.stack(arrays, axis=0)  # (S, P, C)

    # Metric-specific scaling
    if metric.lower().endswith("dice") or metric.lower().endswith("jaccard"):
        A = _ensure_percent_if_prob(A)

    # Mean over seeds -> (P, C)
    return np.nanmean(A, axis=0)


def save_violin(data: np.ndarray, title: str, ylabel: str, filename: str, color: str, out_folder: Path, ylim: Optional[Tuple[float, float]] = None):
    """
    data: (P, C)
    """
    P, C = data.shape
    values = [data[:, c] for c in range(C)]

    fig, ax = plt.subplots(figsize=(9, 5))
    parts = ax.violinplot(values, showmeans=True, showmedians=False, showextrema=True)
    for pc in parts['bodies']:
        pc.set_facecolor(color)
        pc.set_edgecolor("black")
        pc.set_alpha(0.35)
    parts['cbars'].set_color("black")
    parts['cmins'].set_color("black")
    parts['cmaxes'].set_color("black")
    if 'cmeans' in parts:
        parts['cmeans'].set_color("black")

    ax.set_xticks(range(1, C + 1))
    ax.set_xticklabels(CLASS_NAMES[:C], rotation=0)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_folder / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_folder / filename}")


def _compute_common_ylim(metric: str, base: Optional[np.ndarray], post: Optional[np.ndarray]) -> Optional[Tuple[float, float]]:
    """
    Compute a common y-range for violins of the same metric across groups.
    Dice/Jaccard are fixed to [0, 100]. Others use data min/max with padding.
    """
    if metric.lower().endswith("dice") or metric.lower().endswith("jaccard"):
        return (0.0, 100.0)

    stacks = []
    if base is not None:
        stacks.append(base)
    if post is not None:
        stacks.append(post)
    if not stacks:
        return None

    allv = np.concatenate(stacks, axis=0)  # (P_total, C)
    finite = np.isfinite(allv)
    if not finite.any():
        return None

    lo = float(np.nanmin(allv[finite]))
    hi = float(np.nanmax(allv[finite]))
    # Add 5% padding
    rng = hi - lo
    if rng <= 0:
        pad = 1.0 if hi == 0 else 0.05 * abs(hi)
    else:
        pad = 0.05 * rng
    lo = lo - pad
    hi = hi + pad
    # Clamp lower bound at 0 for distance-like metrics if all data >= 0
    if lo >= 0 or float(np.nanmin(allv)) >= 0:
        lo = max(0.0, lo)
    return (lo, hi)


def save_radar(baseline_data: Optional[np.ndarray],
               post_data: Optional[np.ndarray],
               title: str,
               filename: str,
               out_folder: Path):
    """
    baseline_data, post_data: (P, C)
    """
    if baseline_data is None and post_data is None:
        return

    N = baseline_data.shape[1] if baseline_data is not None else post_data.shape[1]
    labels = CLASS_NAMES[:N]
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the loop

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))

    def mean_close(arr: np.ndarray) -> List[float]:
        v = np.nanmean(arr, axis=0).tolist()
        return v + v[:1]

    if baseline_data is not None:
        b = mean_close(baseline_data)
        ax.plot(angles, b, 'o-', linewidth=2, label="Baseline", color=COLORS["baseline"])
        ax.fill(angles, b, color=COLORS["baseline"], alpha=0.25)

    if post_data is not None:
        p = mean_close(post_data)
        ax.plot(angles, p, 'o-', linewidth=2, label="Post-processed", color=COLORS["post"])
        ax.fill(angles, p, color=COLORS["post"], alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    fig.tight_layout()
    out_folder.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_folder / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved {out_folder / filename}")


def split_folders(folders: List[str]) -> Tuple[List[str], List[str]]:
    """
    Heuristic split into baseline vs post-processing groups.
    """
    baseline = [f for f in folders if "baseline" in f.lower()]
    post = [f for f in folders if any(k in f.lower() for k in ["postprocessing", "post-processing", "postproc"])]
    # Fallback: if not detected, assume first half baseline, second half post
    if not baseline or not post:
        n = len(folders)
        half = n // 2
        if not baseline:
            baseline = folders[:half]
        if not post:
            post = folders[half:]
    return baseline, post


def main():
    ap = argparse.ArgumentParser(description="Visualize 3D metrics: Baseline vs Post-processing (radar + violins).")
    ap.add_argument("--folders", type=str, nargs="+", required=True,
                    help="List of folders containing metric files (.npy/.npz). Baseline and post folders mixed.")
    ap.add_argument("--out", type=str, default="metrics_compare", help="Output folder for plots.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_folders, post_folders = split_folders(args.folders)
    print(f"[INFO] Baseline folders: {len(baseline_folders)}")
    for f in baseline_folders: print(f"  - {f}")
    print(f"[INFO] Post folders: {len(post_folders)}")
    for f in post_folders: print(f"  - {f}")

    # Load data per metric
    data = {}
    for m in METRICS:
        base = load_group_data(baseline_folders, m)
        post = load_group_data(post_folders, m)
        data[m] = (base, post)

    # Y-labels per metric
    ylabels = {
        "3d_dice": "DSC (%)",
        "3d_hd95": "HD95 (mm)",
        "3d_jaccard": "Jaccard (%)",
        "3d_assd": "ASSD (mm)",
    }

    # Plots
    for m in METRICS:
        base, post = data[m]
        print(f"[INFO] {m}: baseline={'ok' if base is not None else 'none'}, post={'ok' if post is not None else 'none'}")

        # Common y-limits for both groups for this metric
        ylim = _compute_common_ylim(m, base, post)

        # Violins per group (share same y-axis)
        if base is not None:
            save_violin(base, f"{m} per class (Baseline)", ylabels.get(m, m), f"{m}_violin_baseline.png",
                        color=COLORS["baseline"], out_folder=out_dir, ylim=ylim)
        if post is not None:
            save_violin(post, f"{m} per class (Post-processed)", ylabels.get(m, m), f"{m}_violin_post.png",
                        color=COLORS["post"], out_folder=out_dir, ylim=ylim)

        # Radar Baseline vs Post
        if base is not None and post is not None:
            save_radar(base, post, f"{m} mean per class (Baseline vs Post-processed)",
                       f"{m}_radar_baseline_vs_post.png", out_folder=out_dir)

    print(f"[DONE] All plots saved in: {out_dir}")


if __name__ == "__main__":
    main()