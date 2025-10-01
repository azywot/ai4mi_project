# preprocessing.py
import numpy as np
import scipy.ndimage as ndimage

# ------------------------------
# Core: in-plane resampling only
# ------------------------------
def resample_inplane(volume, spacing_yxz, new_xy=1.0, is_label=False):
    """
    Resample ONLY (y, x) to 'new_xy' mm. Keep z unchanged (slice count preserved).
    volume shape: (y, x, z)
    spacing_yxz: (dy, dx, dz)   # careful: this matches array axes (y, x, z)
    """
    dy, dx, dz = spacing_yxz
    fy = dy / new_xy
    fx = dx / new_xy
    fz = 1.0
    order = 0 if is_label else 1  # NN for labels, linear for CT
    return ndimage.zoom(volume, (fy, fx, fz), order=order)

# ------------------------------
# Intensity preprocessing
# ------------------------------
def hu_window(volume, hu_min=-100, hu_max=400):
    """Clip to thoracic soft-tissue window."""
    return np.clip(volume, hu_min, hu_max)

def minmax01(volume, vmin, vmax):
    """Min–max scale to [0,1] with safe denom."""
    denom = max(vmax - vmin, 1e-6)
    return (volume - vmin) / denom

# ------------------------------
# Full CT / Label pipelines
# ------------------------------
def preprocess_ct_inplane_only(volume, spacing_yxz, hu_min=-100, hu_max=400, target_xy=1.0, do_resample=True):
    """
    Pipeline for CT:
      1) optional in-plane resample to (target_xy, target_xy) mm (keep z)
      2) HU clip to [hu_min, hu_max]
      3) Min–max to [0,1]
    """
    if do_resample:
        volume = resample_inplane(volume, spacing_yxz, new_xy=target_xy, is_label=False)
    volume = hu_window(volume, hu_min, hu_max)
    volume = minmax01(volume, hu_min, hu_max)
    return volume

def preprocess_label_inplane_only(volume, spacing_yxz, target_xy=1.0, do_resample=True):
    """
    Pipeline for labels:
      1) optional in-plane resample to (target_xy, target_xy) mm (keep z)
    No intensity ops on labels.
    """
    if do_resample:
        volume = resample_inplane(volume, spacing_yxz, new_xy=target_xy, is_label=True)
    return volume

# ------------------------------
# Optional: body crop helper
# ------------------------------
def compute_body_bbox_from_hu(volume_hu, hu_thresh=-600, margin=16):
    """
    Compute a single 2D body bounding box (y1:y2, x1:x2) across all slices.
    Pass a HU-clipped volume (e.g., after hu_window).
    """
    mask = (volume_hu > hu_thresh).astype(np.uint8)  # body ~= not air
    proj = mask.max(axis=2)  # collapse over z
    ys, xs = np.where(proj > 0)
    if len(ys) == 0:
        # Fallback: full-frame if body mask fails
        return (0, volume_hu.shape[0], 0, volume_hu.shape[1])
    y1 = max(0, ys.min() - margin)
    y2 = min(volume_hu.shape[0], ys.max() + 1 + margin)
    x1 = max(0, xs.min() - margin)
    x2 = min(volume_hu.shape[1], xs.max() + 1 + margin)
    return (y1, y2, x1, x2)
