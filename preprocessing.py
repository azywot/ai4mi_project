import numpy as np
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes, label, find_objects, zoom

def apply_hu_window(arr, hu_min=-1000, hu_max=1000):
    """Clip CT values to HU window."""
    return np.clip(arr, hu_min, hu_max)

def zscore_normalize(arr):
    """Z-score normalization per volume."""
    mu, sigma = np.mean(arr), np.std(arr)
    return (arr - mu) / (sigma + 1e-8)

def get_body_mask(arr, threshold=None):
    """
    Create a body mask using Otsu threshold (or fixed HU threshold).
    Input arr is already HU-windowed.
    """
    # Compute threshold on central axial slice (last axis = z)
    if threshold is None:
        mid_slice = arr[:, :, arr.shape[2] // 2]
        threshold = threshold_otsu(mid_slice)

    mask = arr > threshold
    # Keep largest connected component
    labeled, n = label(mask)
    if n > 0:
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = counts.argmax()
        mask = (labeled == largest)
    mask = binary_fill_holes(mask)
    return mask.astype(np.uint8)

def crop_to_body(arr, mask, margin=10):
    """
    Crop array to bounding box around mask (+margin).
    Expecting array shape (y, x, z)
    """
    slices = find_objects(mask)
    if slices and len(slices) > 0:
        yslice, xslice, zslice = slices[0]
        ymin, ymax = yslice.start, yslice.stop
        xmin, xmax = xslice.start, xslice.stop
        zmin, zmax = zslice.start, zslice.stop

        # Add margin
        ymin = max(ymin - margin, 0)
        ymax = min(ymax + margin, arr.shape[0])
        xmin = max(xmin - margin, 0)
        xmax = min(xmax + margin, arr.shape[1])
        zmin = max(zmin - margin, 0)
        zmax = min(zmax + margin, arr.shape[2])

        return arr[ymin:ymax, xmin:xmax, zmin:zmax]
    return arr

def resample_inplane(arr, new_spacing=(1.0, 1.0), spacing=(1.0, 1.0, 2.5), is_label=False):
    """
    Resample CT volume in-plane (y,x) to given spacing.
    spacing = (dy, dx, dz) corresponding to array shape (y,x,z)
    new_spacing = (new_dy, new_dx)
    """
    dy, dx, dz = spacing
    new_dy, new_dx = new_spacing

    zoom_factors = [dy / new_dy, dx / new_dx, 1.0]  # keep z unchanged
    order = 0 if is_label else 1
    return zoom(arr, zoom=zoom_factors, order=order)
