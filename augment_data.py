from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------
# Helper types and utilities
# ---------------------------
Pair = Tuple[Tensor, Tensor]  # (image, mask)


def _ensure_chw(x: Tensor) -> Tensor:
    """Ensure tensor is [C,H,W]. Accepts [H,W] or [1,H,W]."""
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected 2D or 3D tensor, got shape {tuple(x.shape)}")


def _onehot_to_labels(mask_oh: Tensor) -> Tensor:
    """mask_oh: [K,H,W] -> labels: [1,H,W] (long)."""
    if mask_oh.dtype == torch.long and mask_oh.ndim == 3 and mask_oh.shape[0] == 1:
        return mask_oh  # already labels shaped [1,H,W]
    if mask_oh.ndim != 3:
        raise ValueError("mask must be [K,H,W] or [1,H,W] labels")
    labels = mask_oh.argmax(dim=0, keepdim=True).to(torch.long)
    return labels


def _labels_to_onehot(labels: Tensor, K: int) -> Tensor:
    """labels: [1,H,W] (long) -> [K,H,W] (float)
    Uses float32 so it can be interpolated if needed (we still keep nearest for masks).
    """
    if labels.ndim != 3 or labels.shape[0] != 1:
        raise ValueError("labels must be [1,H,W]")
    H, W = labels.shape[-2:]
    oh = F.one_hot(labels.squeeze(0), num_classes=K).permute(2, 0, 1).to(torch.float32)
    return oh


# ---------------------------------
# Transform base and composition API
# ---------------------------------
class PairTransform:
    """Base class for paired transforms (image + mask)."""

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        raise NotImplementedError


class Compose(PairTransform):
    def __init__(self, transforms: Iterable[PairTransform]):
        self.transforms = list(transforms)

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask


# ---------------------
# Deterministic sampling
# ---------------------
class RNG:
    """Small wrapper around torch.Generator to make ops deterministic per call."""

    def __init__(
        self, seed: Optional[int] = None, device: Optional[torch.device] = None
    ):
        self.device = device or torch.device("cpu")
        self.gen = torch.Generator(device=self.device)
        if seed is not None:
            self.gen.manual_seed(int(seed))

    def randint(self, low: int, high: int) -> int:
        return int(
            torch.randint(
                low, high, (1,), generator=self.gen, device=self.device
            ).item()
        )

    def rand(self) -> float:
        return float(torch.rand((), generator=self.gen, device=self.device).item())

    def normal(self, shape, mean=0.0, std=1.0) -> Tensor:
        return torch.normal(
            mean=mean, std=std, size=shape, generator=self.gen, device=self.device
        )


# -----------------------
# Individual transformations
# -----------------------
class RandomFlipRotate(PairTransform):
    def __init__(
        self,
        p_flip_h: float = 0.5,
        p_flip_v: float = 0.5,
        max_rotate: float = 15.0,
        rng: Optional[RNG] = None,
    ):
        self.p_flip_h = p_flip_h
        self.p_flip_v = p_flip_v
        self.max_rotate = max_rotate  # degrees
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        image = _ensure_chw(image)
        mask = _ensure_chw(mask)
        # Horizontal flip
        if self.rng.rand() < self.p_flip_h:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[2])
        # Vertical flip
        if self.rng.rand() < self.p_flip_v:
            image = torch.flip(image, dims=[1])
            mask = torch.flip(mask, dims=[1])
        # Rotation
        angle = (self.rng.rand() * 2 - 1) * self.max_rotate  # in [-max,+max]
        if abs(angle) > 1e-3:
            theta = math.radians(angle)
            c, s = math.cos(theta), math.sin(theta)
            # affine grid expects [N,2,3]
            A = torch.tensor(
                [[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=image.device
            ).unsqueeze(0)
            H, W = image.shape[-2:]
            grid = F.affine_grid(A, size=(1, image.shape[0], H, W), align_corners=False)
            image = F.grid_sample(
                image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
            # Masks: nearest
            mask = F.grid_sample(
                mask.unsqueeze(0),
                grid,
                mode="nearest",
                padding_mode="zeros",
                align_corners=False,
            ).squeeze(0)
        return image, mask


class IntensityShift(PairTransform):
    """Random multiplicative + additive intensity change on the image only.
    Works on images assumed to be in [0,1]; clamps back to [0,1].
    """

    def __init__(
        self, mul_range=(0.9, 1.1), add_range=(-0.05, 0.05), rng: Optional[RNG] = None
    ):
        self.mul_range = mul_range
        self.add_range = add_range
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        image = _ensure_chw(image)
        mask = _ensure_chw(mask)
        m = self.mul_range[0] + self.rng.rand() * (
            self.mul_range[1] - self.mul_range[0]
        )
        a = self.add_range[0] + self.rng.rand() * (
            self.add_range[1] - self.add_range[0]
        )
        image = torch.clamp(image * m + a, 0.0, 1.0)
        return image, mask


class HUWindowing(PairTransform):
    """Apply CT HU windowing to the image.

    Input image is expected in HU units (float32). If your pipeline currently normalizes
    to [0,1] earlier, apply this transform *before* that normalization, or
    set `input_hu_range` to map back to HU-like values first.

    Args:
        center: window center (e.g., 40 for soft tissue, -600 for lung)
        width: window width (e.g., 400 for soft tissue, 1500 for lung)
        input_hu_range: if not None, assume input is in [0,1] that corresponds to this HU range
                        (minHU, maxHU); we linearly map to HU, window, then rescale to [0,1].
    """

    def __init__(
        self,
        center: float = 40.0,
        width: float = 400.0,
        input_hu_range: Optional[Tuple[float, float]] = None,
    ):
        self.center = float(center)
        self.width = float(width)
        self.input_hu_range = input_hu_range

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        image = _ensure_chw(image).to(torch.float32)
        mask = _ensure_chw(mask)
        if self.input_hu_range is not None:
            lo, hi = self.input_hu_range
            image_hu = image * (hi - lo) + lo
        else:
            image_hu = image
        wmin = self.center - self.width / 2.0
        wmax = self.center + self.width / 2.0
        image_hu = torch.clamp(image_hu, wmin, wmax)
        image_w = (image_hu - wmin) / max(self.width, 1e-6)  # -> [0,1]
        return image_w, mask


class ElasticDeformation(PairTransform):
    """Elastic deformation using a smoothed random displacement field.

    Based on Simard et al. (2003). Implemented with grid_sample.

    Args:
        alpha: scaling of the displacement magnitude (pixels)
        sigma: Gaussian smoothing (std in pixels) of the displacement
        rng: RNG for reproducibility
    """

    def __init__(
        self, alpha: float = 30.0, sigma: float = 4.0, rng: Optional[RNG] = None
    ):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.rng = rng or RNG()

    @staticmethod
    def _gaussian_kernel2d(sigma: float, device) -> Tensor:
        # kernel size: 6*sigma rounded up to nearest odd
        k = max(3, int(6 * sigma) | 1)
        ax = torch.arange(k, device=device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum().clamp_min(1e-8)
        return kernel

    def __call__(self, image: Tensor, mask: Tensor) -> Pair:
        image = _ensure_chw(image)
        mask = _ensure_chw(mask)
        C, H, W = image.shape
        device = image.device
        # random displacement (2,H,W)
        disp = self.rng.normal((2, H, W))
        disp = disp.to(device, dtype=torch.float32)
        # smooth with gaussian
        kernel = self._gaussian_kernel2d(self.sigma, device=device)
        kernel = kernel.expand(2, 1, -1, -1)
        disp = F.conv2d(
            disp.unsqueeze(0), kernel, padding=kernel.shape[-1] // 2, groups=2
        ).squeeze(0)
        # scale
        disp = disp * (self.alpha / max(H, W))  # roughly alpha pixels in magnitude
        # build base grid in [-1,1]
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)  # [H,W,2]
        # convert disp (in normalized coords). disp is in pixels -> normalize
        disp_x = disp[1] * 2.0  # approx; scaling above already normalized by size
        disp_y = disp[0] * 2.0
        grid = grid + torch.stack((disp_x, disp_y), dim=-1)
        grid = grid.clamp(-1, 1)
        # sample
        image_d = F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        mask_d = F.grid_sample(
            mask.unsqueeze(0),
            grid.unsqueeze(0),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        return image_d, mask_d


# ---------------------------
# Public API: presets & build
# ---------------------------
@dataclass
class AugPreset:
    name: str
    builder: Callable[[Optional[int]], PairTransform]
    note: str = ""


def _preset_none(seed: Optional[int]) -> PairTransform:
    return Compose([])


def _preset_basic(seed: Optional[int]) -> PairTransform:
    rng = RNG(seed)
    return Compose(
        [
            RandomFlipRotate(p_flip_h=0.5, p_flip_v=0.5, max_rotate=15.0, rng=rng),
            IntensityShift(mul_range=(0.9, 1.1), add_range=(-0.05, 0.05), rng=rng),
        ]
    )


def _preset_basic_elastic(seed: Optional[int]) -> PairTransform:
    rng = RNG(seed)
    return Compose(
        [
            RandomFlipRotate(p_flip_h=0.5, p_flip_v=0.5, max_rotate=15.0, rng=rng),
            IntensityShift(mul_range=(0.9, 1.1), add_range=(-0.05, 0.05), rng=rng),
            ElasticDeformation(alpha=30.0, sigma=4.0, rng=rng),
        ]
    )


def _preset_hu_basic(seed: Optional[int]) -> PairTransform:
    rng = RNG(seed)
    return Compose(
        [
            HUWindowing(center=40.0, width=400.0, input_hu_range=None),
            RandomFlipRotate(p_flip_h=0.5, p_flip_v=0.5, max_rotate=15.0, rng=rng),
            IntensityShift(mul_range=(0.95, 1.05), add_range=(-0.02, 0.02), rng=rng),
        ]
    )


PRESETS: Dict[str, AugPreset] = {
    "none": AugPreset("none", _preset_none, note="No augmentation"),
    "basic": AugPreset("basic", _preset_basic, note="flip+rotation+intensity"),
    "basic+elastic": AugPreset(
        "basic+elastic", _preset_basic_elastic, note="basic + elastic deformation"
    ),
    "hu+basic": AugPreset("hu+basic", _preset_hu_basic, note="HU window -> basic"),
}


def build_augmentation(name: str, seed: Optional[int] = None) -> PairTransform:
    """Build augmentation pipeline by preset name."""
    key = name.lower()
    if key not in PRESETS:
        raise KeyError(
            f"Unknown augmentation preset '{name}'. Available: {list(PRESETS)}"
        )
    return PRESETS[key].builder(seed)


# -------------------------------------
# Dataset wrapper to apply augmentations
# -------------------------------------
class AugmentedDataset(torch.utils.data.Dataset):
    """Wrap an existing dataset to apply paired augmentations.

    Assumptions on the wrapped dataset's __getitem__ output:
        item is a dict with keys 'images' -> Tensor[C,H,W] (float)
                               'gts'    -> Tensor[K,H,W] one-hot OR [1,H,W] labels
        Any additional keys are preserved.
    """

    def __init__(self, base_ds, transform: PairTransform):
        self.base = base_ds
        self.transform = transform
        # Try to detect number of classes K from first sample
        ex = self.base[0]
        mask = ex["gts"]
        K = mask.shape[0] if mask.ndim == 3 else None
        self.num_classes = K

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        image: Tensor = item["images"].to(torch.float32)
        mask: Tensor = item["gts"]
        # standardize mask to [K,H,W] float
        if mask.ndim == 3 and mask.shape[0] > 1:
            mask_oh = mask.to(torch.float32)
        else:
            labels = _onehot_to_labels(mask)  # handles [1,H,W] long
            if self.num_classes is None:
                raise RuntimeError(
                    "num_classes could not be inferred; provide one-hot masks in the dataset"
                )
            mask_oh = _labels_to_onehot(labels, self.num_classes)
        img_t, msk_t = self.transform(image, mask_oh)
        # keep one-hot format for downstream criterion
        out = dict(item)
        out["images"] = img_t
        out["gts"] = msk_t
        return out


import torchvision.utils as vutils
from pathlib import Path
from PIL import Image


def save_aug_examples(dataset, dest: Path, n_samples: int = 4, n_augments: int = 3):
    """
    Save example augmentations from an AugmentedDataset.

    Args:
        dataset: an AugmentedDataset (or any dataset yielding dict with "images" and "gts")
        dest: output folder (Path)
        n_samples: number of different dataset items
        n_augments: how many augmentations to draw per sample
    """
    dest = Path(dest) / "aug_examples"
    dest.mkdir(parents=True, exist_ok=True)

    for idx in range(min(n_samples, len(dataset))):
        for j in range(n_augments):
            item = dataset[idx]
            img = item["images"]
            # Convert image tensor [C,H,W] to grid
            grid = vutils.make_grid(img.unsqueeze(0), normalize=True, scale_each=True)
            ndarr = grid.mul(255).byte().permute(1, 2, 0).cpu().numpy()
            Image.fromarray(ndarr).save(dest / f"sample{idx:02d}_aug{j:02d}.png")


def add_augmentation_cli(parser) -> None:
    """Extend an argparse.ArgumentParser with augmentation flags."""
    group = parser.add_argument_group("Augmentation")
    group.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=list(PRESETS.keys()),
        help="Augmentation preset to use for training",
    )
    group.add_argument(
        "--aug-seed",
        type=int,
        default=None,
        help="Seed for deterministic augmentation",
    )


def wrap_train_dataset_if_needed(train_set, args):
    """Convenience helper to wrap train_set with augmentation based on args.
    Expects args.aug and args.aug_seed (added by add_augmentation_cli)."""
    if getattr(args, "aug", "none") and args.aug != "none":
        aug = build_augmentation(args.aug, seed=args.aug_seed)
        return AugmentedDataset(train_set, aug)
    return train_set
