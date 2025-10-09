import numpy as np
import cv2

def _percentile_window_u8(volume: np.ndarray, p_low=1.0, p_high=99.5,
                          hu_clip=(-1000, 1500)) -> np.ndarray:
    v = np.clip(volume.astype(np.float32), hu_clip[0], hu_clip[1])
    body = v > -800
    vals = v[body] if body.any() else v.ravel()
    lo = float(np.percentile(vals, p_low))
    hi = float(np.percentile(vals, p_high))
    if hi <= lo:
        lo, hi = -150.0, 300.0
    v = np.clip(v, lo, hi)
    v = (v - lo) / (hi - lo + 1e-6)
    return (v * 255.0).astype(np.uint8)

def _tone_map_log_u8(img_u8: np.ndarray, alpha: float = 3.0) -> np.uint8:
    x = img_u8.astype(np.float32) / 255.0
    y = np.log1p(alpha * x) / np.log1p(alpha)
    return np.clip(y * 255.0, 0, 255).astype(np.uint8)

def _clahe_adaptive_u8(img_u8: np.ndarray, base_clip=1.85, tile=(8, 8)) -> np.uint8:
    p10, p90 = np.percentile(img_u8, (10, 90))
    spread = max((p90 - p10) / 255.0, 1e-3)
    clip = float(np.clip(base_clip * (0.9 + 1.2 * spread), 1.4, 2.6))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    return clahe.apply(img_u8)

def _z_smooth_3tap(vol_u8: np.ndarray, w=(0.2, 0.6, 0.2)) -> np.ndarray:
    if vol_u8.shape[2] < 3:
        return vol_u8
    out = vol_u8.astype(np.float32).copy()
    Z = vol_u8.shape[2]
    for k in range(1, Z - 1):
        out[:, :, k] = w[0]*vol_u8[:, :, k-1] + w[1]*vol_u8[:, :, k] + w[2]*vol_u8[:, :, k+1]
    return np.clip(out, 0, 255).astype(np.uint8)

def _unsharp_gated(img_u8: np.uint8, sigma=1.0, amount=0.14) -> np.uint8:
    src = img_u8.astype(np.float32)
    blur = cv2.GaussianBlur(src, (0, 0), sigmaX=sigma, sigmaY=sigma)
    mask = src - blur
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    mag = mag / (mag.max() + 1e-6)
    gate = np.sqrt(mag)  
    out = src + (amount * gate) * mask
    return np.clip(out, 0, 255).astype(np.uint8)

def preprocess_ct_volume(volume: np.ndarray,
                         p_low=1.0, p_high=99.5, hu_clip=(-1000, 1500),
                         alpha=3.0, base_clip=1.85,
                         z_smooth=True, z_w=(0.2, 0.6, 0.2),
                         unsharp_sigma=1.0, unsharp_amount=0.14) -> np.ndarray:

    vol = _percentile_window_u8(volume, p_low=p_low, p_high=p_high, hu_clip=hu_clip)

    Z = vol.shape[2]
    enh = np.zeros_like(vol, dtype=np.uint8)
    for z in range(Z):
        s = _tone_map_log_u8(vol[:, :, z], alpha=alpha)
        s = _clahe_adaptive_u8(s, base_clip=base_clip, tile=(8, 8))
        enh[:, :, z] = s

    if z_smooth:
        enh = _z_smooth_3tap(enh, w=z_w)

    out = np.zeros_like(enh, dtype=np.uint8)
    for z in range(Z):
        out[:, :, z] = _unsharp_gated(enh[:, :, z], sigma=unsharp_sigma, amount=unsharp_amount)
    return out