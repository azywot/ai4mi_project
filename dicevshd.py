#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import glob
import numpy as np
import matplotlib.pyplot as plt
import itertools


def _load_np(path: Path) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=True)
    if path.suffix == ".npz":
        z = np.load(path, allow_pickle=True)
        for k in ["data", "arr_0", "values", "metric", "A"]:
            if k in z.files and isinstance(z[k], np.ndarray):
                return z[k]
        for k in z.files:
            if isinstance(z[k], np.ndarray):
                return z[k]
    return None


def _try_load_metric(dir_: Path, metric: str) -> Optional[np.ndarray]:
    candidates = [
        dir_ / f"{metric}.npy",
        dir_ / f"{metric}.npz",
        dir_ / "metrics" / f"{metric}.npy",
        dir_ / "metrics" / f"{metric}.npz",
    ]
    for p in candidates:
        arr = _load_np(p)
        if arr is not None:
            return arr
    return None


def _try_load_case_ids(dir_: Path) -> Optional[List[str]]:
    for fname in ["case_ids.npy", "case_ids.npz", "cases.npy", "ids.npy"]:
        for p in [dir_ / fname, dir_ / "metrics" / fname]:
            if p.exists():
                if p.suffix == ".npy":
                    a = np.load(p, allow_pickle=True)
                    return [str(x) for x in a.tolist()]
                else:
                    z = np.load(p, allow_pickle=True)
                    for k in z.files:
                        return [str(x) for x in z[k].tolist()]
    return None


def _expand_paths(parts: List[str]) -> List[Path]:
    paths: List[Path] = []
    for item in parts:
        for p in glob.glob(item):
            paths.append(Path(p))
    seen, unique = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    return unique


def parse_groups(group_args: List[str]) -> List[Tuple[str, List[Path]]]:
    groups: List[Tuple[str, List[Path]]] = []
    for g in group_args:
        if ":" not in g:
            raise ValueError(f"--group expects NAME:paths, got: {g}")
        name, paths_str = g.split(":", 1)
        parts = [s.strip() for s in paths_str.split(",") if s.strip()]
        dirs = _expand_paths(parts)
        if not dirs:
            raise ValueError(f"No directories resolved for group '{name}' from: {paths_str}")
        groups.append((name, dirs))
    return groups


def _normalize_metric_array(arr: np.ndarray, seed_dir: Path, metric: str) -> np.ndarray:
    # Ensure (patients, classes)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
        elif arr.ndim != 2:
            raise ValueError(f"Unexpected shape for {metric} in {seed_dir}: {arr.shape}")
    # If it looks transposed (classes, patients), fix
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 6 and arr.shape[1] > 6:
        arr = arr.T
    return arr


def stack_seeds(metric: str,
                seed_dirs: List[Path],
                align: str = "union",
                debug: bool = False) -> Tuple[np.ndarray, Optional[List[str]]]:
    """
    Returns:
      A: (S, P, C) aligned across seeds (NaNs for missing cases if union/intersect with IDs)
      ids: resolved patient IDs (len P) or None if IDs unavailable and align='min'
    """
    assert align in ("min", "union", "intersect")

    arrays: List[np.ndarray] = []
    id_lists: List[Optional[List[str]]] = []
    for d in seed_dirs:
        arr = _try_load_metric(d, metric)
        if arr is None:
            print(f"[WARN] Missing {metric} in {d}")
            continue
        arr = _normalize_metric_array(arr, d, metric)
        arrays.append(arr)
        id_lists.append(_try_load_case_ids(d))

    if not arrays:
        raise RuntimeError(f"No arrays found for metric {metric} in any of: {seed_dirs}")

    if debug:
        for d, a, ids in zip(seed_dirs, arrays, id_lists):
            print(f"[DBG] {metric} in {d} -> shape={a.shape}, ids={'yes' if ids else 'no'}")

    ids_available = all(ids is not None for ids in id_lists)
    if not ids_available or align == "min":
        min_pat = min(a.shape[0] for a in arrays)
        if any(a.shape[0] != min_pat for a in arrays):
            print(f"[INFO] Aligning {metric} arrays to min patients={min_pat} (no IDs or --align=min)")
        arrays = [a[:min_pat] for a in arrays]
        A = np.stack(arrays, axis=0)
        return A, None

    # Align by patient IDs across seeds
    sets = [set(ids) for ids in id_lists]  # type: ignore
    target_ids = sorted(set.union(*sets)) if align == "union" else sorted(set.intersection(*sets))
    if debug:
        print(f"[DBG] {metric}: align={align}, patients kept={len(target_ids)}")

    # Ensure same class count
    C = arrays[0].shape[1]
    for a in arrays:
        if a.shape[1] != C:
            raise ValueError(f"Inconsistent class count for {metric}: {C} vs {a.shape[1]}")

    idx_maps = [{pid: i for i, pid in enumerate(ids)} for ids in id_lists]  # type: ignore
    S, P = len(arrays), len(target_ids)
    A = np.full((S, P, C), np.nan, dtype=float)
    for s, (arr, id2row) in enumerate(zip(arrays, idx_maps)):
        for p, pid in enumerate(target_ids):
            i = id2row.get(pid, None)
            if i is not None:
                A[s, p, :] = arr[i, :]
    return A, target_ids


def ensure_dice_percent(x: np.ndarray) -> np.ndarray:
    finite = np.isfinite(x)
    if finite.any():
        m, M = np.nanmin(x[finite]), np.nanmax(x[finite])
        if 0.0 <= m and M <= 1.5:
            return 100.0 * x
    return x


def seedmean_per_patient(A: np.ndarray) -> np.ndarray:
    # A: (S, P, C) -> (P, C) NaN-safe mean across seeds
    return np.nanmean(A, axis=0)


def avg_over_classes(X: np.ndarray, include_bg: bool = True) -> np.ndarray:
    # X: (P, C) -> (P,)
    P, C = X.shape
    if include_bg or C == 1:
        sl = slice(0, C)
    else:
        sl = slice(1, C)
    return np.nanmean(X[:, sl], axis=1)


def build_group_points(seed_dirs: List[Path],
                       include_bg: bool = True,
                       align: str = "union",
                       debug: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    A_dice, ids_d = stack_seeds("3d_dice", seed_dirs, align=align, debug=debug)
    A_hd95, ids_h = stack_seeds("3d_hd95", seed_dirs, align=align, debug=debug)

    # Align patients between metrics by ID if available
    if ids_d is not None and ids_h is not None:
        idx_h = {pid: i for i, pid in enumerate(ids_h)}
        common = [pid for pid in ids_d if pid in idx_h]
        if debug:
            print(f"[DBG] Dice∩HD95 common patients: {len(common)}")
        idx_d_list = [i for i, pid in enumerate(ids_d) if pid in idx_h]
        idx_h_list = [idx_h[pid] for pid in common]
        A_dice = A_dice[:, idx_d_list, :]
        A_hd95 = A_hd95[:, idx_h_list, :]
        ids = common
    else:
        P = min(A_dice.shape[1], A_hd95.shape[1])
        A_dice, A_hd95 = A_dice[:, :P], A_hd95[:, :P]
        ids = [f"case_{i:03d}" for i in range(P)]

    dice_ppc = seedmean_per_patient(A_dice)
    hd95_ppc = seedmean_per_patient(A_hd95)

    dice_p = avg_over_classes(dice_ppc, include_bg=include_bg)
    hd95_p = avg_over_classes(hd95_ppc, include_bg=include_bg)
    dice_p = ensure_dice_percent(dice_p)

    # Drop NaNs if a patient is missing in all seeds for a metric
    mask = np.isfinite(dice_p) & np.isfinite(hd95_p)
    dice_p = dice_p[mask]
    hd95_p = hd95_p[mask]
    ids = [pid for pid, keep in zip(ids, mask) if keep]

    return dice_p, hd95_p, ids


def make_plot(
    groups: List[Tuple[str, List[Path]]],
    out_path: Path,
    include_bg: bool = True,
    connect_patients: bool = False,
    xlim: Tuple[float, float] = (50, 100),
    ylim: Tuple[float, float] = (0, 100),
    align: str = "union",
    debug: bool = False,
    show_avg: bool = True,
):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 7))

    # Fixed palette: 1) Baseline yellow, 2) Full blue, 3) Pink, then others
    colors = ["#FDBF00", "#1f77b4", "#e377c2", "#2ca02c", "#9467bd", "#8c564b"]
    markers = ["o", "^", "s", "D", "v", "P"]
    color_cycle = itertools.cycle(colors)
    marker_cycle = itertools.cycle(markers)

    group_data = []
    common_ids = None
    for (name, seed_dirs) in groups:
        dice_p, hd95_p, ids = build_group_points(seed_dirs, include_bg=include_bg, align=align, debug=debug)
        if debug:
            print(f"[DBG] Group '{name}': n_points={len(ids)}")
        group_data.append((name, dice_p, hd95_p, ids))
        s = set(ids)
        common_ids = s if common_ids is None else (common_ids & s)

    styles: Dict[str, Tuple[str, str]] = {}
    for name, dice_p, hd95_p, _ in group_data:
        c = next(color_cycle)
        m = next(marker_cycle)
        styles[name] = (c, m)
        ax.scatter(dice_p, hd95_p, s=42, c=c, marker=m, alpha=0.9, label=name, edgecolor="none")

    # Group-average stars (black outline)
    if show_avg:
        for name, dice_p, hd95_p, _ in group_data:
            c, _ = styles[name]
            mu_d = float(np.nanmean(dice_p))
            mu_h = float(np.nanmean(hd95_p))
            ax.scatter([mu_d], [mu_h], s=240, marker="*", c=c, edgecolor="black", linewidths=1.4, zorder=6)

    ax.set_xlabel("DSC (%)")
    ax.set_ylabel("HD95 (mm)")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend(loc="upper left", frameon=True)

    # Optional connectors
    if connect_patients and common_ids and len(groups) >= 2:
        order = [g[0] for g in groups]
        coords: Dict[str, Dict[str, Tuple[float, float]]] = {}
        for name, dice_p, hd95_p, ids in group_data:
            coords[name] = {pid: (float(d), float(h)) for pid, d, h in zip(ids, dice_p, hd95_p)}
        for pid in sorted(common_ids):
            pts = [coords[name][pid] for name in order if pid in coords[name]]
            if len(pts) >= 2:
                xs, ys = zip(*pts)
                ax.plot(xs, ys, color="#9aa0a6", lw=0.9, ls="--", alpha=0.6)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved → {out_path.with_suffix('.pdf')} and .png")


def main():
    ap = argparse.ArgumentParser(
        description="DSC-vs-HD95 per patient, seed-averaged from .npy/.npz metrics (keeps class 0)."
    )
    ap.add_argument(
        "--group",
        action="append",
        default=[],
        help='Model group as "Name:/dir/seed42/metrics,/dir/seed37/metrics" (supports globs). Repeatable.',
    )
    ap.add_argument("--out", type=Path, required=True, help="Output path without extension.")
    ap.add_argument("--connect", action="store_true", help="Draw dashed connectors for same patient across groups.")
    ap.add_argument("--xlim", type=float, nargs=2, default=(50, 100))
    ap.add_argument("--ylim", type=float, nargs=2, default=(0, 100))
    ap.add_argument("--align", type=str, choices=["min", "intersect", "union"], default="union",
                    help="Patient alignment across seeds: 'union' keeps all available patients (NaN-safe).")
    ap.add_argument("--debug", action="store_true", help="Print per-seed shapes and resolved counts.")
    ap.add_argument("--no_avg", action="store_true", help="Hide star markers for per-group patient averages.")
    args = ap.parse_args()

    groups = parse_groups(args.group)
    if not groups:
        raise SystemExit("Provide at least one --group")

    make_plot(
        groups=groups,
        out_path=args.out,
        include_bg=True,
        connect_patients=args.connect,
        xlim=tuple(args.xlim),
        ylim=tuple(args.ylim),
        align=args.align,
        debug=args.debug,
        show_avg=not args.no_avg,
    )


if __name__ == "__main__":
    main()