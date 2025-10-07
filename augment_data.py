#!/usr/bin/env python3
from __future__ import annotations

# augment_data.py — unified training + preview augmentations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image

# -------------------------
# batchgenerators (preview)
# -------------------------
from batchgenerators.dataloading.single_threaded_augmenter import (
    SingleThreadedAugmenter,
)
from batchgenerators.transforms.abstract_transforms import Compose as BGCompose
from batchgenerators.transforms.spatial_transforms import (
    MirrorTransform,
    SpatialTransform,
)
from batchgenerators.transforms.color_transforms import (
    ContrastAugmentationTransform,
    GammaTransform,
    BrightnessMultiplicativeTransform,
)
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform

# -------------------------
# project imports
# -------------------------
import torch
import torch.nn.functional as F
from torch import Tensor

from dataset import (
    SliceDataset,
)  # expects dict with 'images' [1,H,W] and 'gts' (one-hot [K,H,W] or labels [1,H,W])
from utils import class2one_hot  # used if labels are not one-hot

# =============================================================================
# Helpers (common)
# =============================================================================


def _to_chw_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 2:
            x = x.unsqueeze(0)
        x = x.to(torch.float32).numpy()
    else:
        x = np.asarray(x, dtype=np.float32)
        if x.ndim == 2:
            x = x[None]
    return x


def _save_png(img_chw: np.ndarray, out_path: Path) -> None:
    assert img_chw.ndim == 3
    if img_chw.shape[0] == 1:
        arr = np.clip(img_chw[0], 0, 1) * 255.0
        Image.fromarray(arr.astype(np.uint8)).save(out_path)
    else:
        arr = np.clip(np.transpose(img_chw, (1, 2, 0)), 0, 1) * 255.0
        Image.fromarray(arr.astype(np.uint8)).save(out_path)


def _colorize_labels(one_hot: np.ndarray) -> np.ndarray:
    K, H, W = one_hot.shape
    labels = np.argmax(one_hot, axis=0)
    lut = np.array(
        [
            [0, 0, 0],  # 0 bg
            [255, 0, 0],  # 1 esophagus
            [0, 255, 0],  # 2 heart
            [0, 0, 255],  # 3 trachea
            [255, 255, 0],  # 4 aorta
        ],
        dtype=np.uint8,
    )
    if K > len(lut):
        extra = np.random.RandomState(0).randint(
            0, 255, size=(K - len(lut), 3), dtype=np.uint8
        )
        lut = np.concatenate([lut, extra], axis=0)
    rgb = lut[labels]  # [H,W,3]
    rgb = np.transpose(rgb, (2, 0, 1)).astype(np.float32) / 255.0  # [3,H,W]
    return rgb


# =============================================================================
# Batchgenerators PREVIEW pipeline (no change to training)
# =============================================================================


class BGDictParser:
    def __init__(self, imgs: np.ndarray, segs: np.ndarray):
        self.data = imgs.astype(np.float32)  # [B,1,H,W]
        self.seg = segs.astype(np.float32)  # [B,K,H,W]
        self.thread_id = 0

    def __iter__(self):
        return self

    def __next__(self):
        return {"data": self.data, "seg": self.seg}

    def set_thread_id(self, thread_id):
        self.thread_id = thread_id


def build_bg_compose(
    *,
    mirror: bool,
    rotations: bool,
    scaling: bool,
    elastic: bool,
    brightness: bool,
    contrast: bool,
    gamma: bool,
    gaussian_noise: bool,
    p_per_sample: float = 0.15,
    patch_shape: Tuple[int, int] = (512, 512),
) -> BGCompose:
    tr: List[Any] = []
    if mirror:
        tr.append(MirrorTransform(axes=(0, 1)))
    if contrast:
        tr.append(
            ContrastAugmentationTransform(
                contrast_range=(0.3, 3.0),
                preserve_range=True,
                per_channel=True,
                p_per_sample=p_per_sample,
            )
        )
    if brightness:
        tr.append(
            BrightnessMultiplicativeTransform(
                multiplier_range=(0.5, 2.0), per_channel=True, p_per_sample=p_per_sample
            )
        )
    if gamma:
        tr.append(
            GammaTransform(
                gamma_range=(0.7, 1.5),
                invert_image=False,
                per_channel=True,
                retain_stats=True,
                p_per_sample=p_per_sample,
            )
        )
    if gaussian_noise:
        tr.append(
            GaussianNoiseTransform(
                noise_variance=(0.0, 0.05), p_per_sample=p_per_sample
            )
        )
    if rotations or scaling or elastic:
        tr.append(
            SpatialTransform(
                patch_size=list(patch_shape),
                patch_center_dist_from_border=[ps // 2 for ps in patch_shape],
                do_elastic_deform=elastic,
                alpha=(0.0, 900.0),
                sigma=(9.0, 13.0),
                do_rotation=rotations,
                angle_x=None,
                angle_y=None,
                angle_z=(-15.0 / 360 * 2 * np.pi, 15.0 / 360 * 2 * np.pi),
                do_scale=scaling,
                scale=(0.85, 1.25),
                border_mode_data="nearest",
                border_cval_data=0.0,
                border_mode_seg="nearest",
                border_cval_seg=0,
                order_data=3,
                order_seg=0,
                p_el_per_sample=p_per_sample,
                p_rot_per_sample=p_per_sample,
                p_scale_per_sample=p_per_sample,
                random_crop=False,
            )
        )
    return BGCompose(tr)


def preset_compose(name: str, patch_shape: Tuple[int, int]) -> BGCompose:
    name = name.lower()
    if name == "dkfz-like":
        return build_bg_compose(
            mirror=True,
            rotations=True,
            scaling=True,
            elastic=True,
            brightness=True,
            contrast=True,
            gamma=True,
            gaussian_noise=True,
            p_per_sample=0.15,
            patch_shape=patch_shape,
        )
    elif name == "basic":
        return build_bg_compose(
            mirror=True,
            rotations=True,
            scaling=False,
            elastic=False,
            brightness=True,
            contrast=False,
            gamma=False,
            gaussian_noise=False,
            p_per_sample=0.15,
            patch_shape=patch_shape,
        )
    elif name == "none":
        return BGCompose([])
    else:
        raise ValueError(f"Unknown preset '{name}'")


class BGDataAugmentor:
    def __init__(
        self,
        dataset_name: str,
        split: str,
        root: Path,
        patch_shape: Optional[Tuple[int, int]] = None,
    ):
        root = Path(root)
        self.ds = SliceDataset(
            split, root, img_transform=None, gt_transform=None, debug=False
        )
        first = self.ds[0]
        img = _to_chw_numpy(first["images"])  # [1,H,W]
        H, W = img.shape[-2:]
        self.patch_shape = patch_shape or (H, W)

    def load_batch(
        self, n: int, K: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        B = min(n, len(self.ds))
        imgs, segs = [], []
        for i in range(B):
            it = self.ds[i]
            img = _to_chw_numpy(it["images"])
            gt = it["gts"]
            if isinstance(gt, torch.Tensor):
                if gt.ndim == 3 and gt.shape[0] > 1:
                    seg = gt.detach().cpu().to(torch.float32).numpy()
                else:
                    if K is None:
                        K = 5
                    lbl = gt.detach().cpu().to(torch.long)  # [1,H,W]
                    seg = (
                        class2one_hot(lbl[None, ...], K=K)[0].to(torch.float32).numpy()
                    )
            else:
                gt = np.asarray(gt)
                if gt.ndim == 3 and gt.shape[0] > 1:
                    seg = gt.astype(np.float32)
                else:
                    if K is None:
                        K = 5
                    onehot = np.eye(K, dtype=np.float32)[gt.astype(np.int64)]
                    seg = np.transpose(onehot, (2, 0, 1))
            imgs.append(img)
            segs.append(seg)
        return np.stack(imgs, 0), np.stack(segs, 0)

    def run(
        self, compose: BGCompose, samples: int, cycles: int = 1
    ) -> Dict[str, np.ndarray]:
        imgs, segs = self.load_batch(samples)
        parser = BGDictParser(imgs, segs)
        aug = SingleThreadedAugmenter(parser, compose)
        outs_data, outs_seg = [], []
        for _ in range(max(1, cycles)):
            out = next(aug)
            outs_data.append(out["data"])
            outs_seg.append(out["seg"])
        return {
            "data": np.concatenate(outs_data, 0),
            "seg": np.concatenate(outs_seg, 0),
        }


# =============================================================================
# PyTorch TRAINING pipeline (used by main.py)
# =============================================================================


def _ensure_chw(x: Tensor) -> Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    return x


def _labels_to_onehot(labels: Tensor, K: int) -> Tensor:
    assert labels.ndim == 3 and labels.shape[0] == 1
    oh = F.one_hot(labels.squeeze(0), num_classes=K).permute(2, 0, 1).to(torch.float32)
    return oh


class PairTransform:
    def __call__(self, image: Tensor, mask_oh: Tensor) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError


class ComposePT(PairTransform):
    def __init__(self, transforms: List[PairTransform]):
        self.transforms = transforms

    def __call__(self, image: Tensor, mask_oh: Tensor) -> Tuple[Tensor, Tensor]:
        for t in self.transforms:
            image, mask_oh = t(image, mask_oh)
        return image, mask_oh


class RNG:
    def __init__(
        self, seed: Optional[int] = None, device: Optional[torch.device] = None
    ):
        self.gen = torch.Generator(device=device or torch.device("cpu"))
        if seed is not None:
            self.gen.manual_seed(int(seed))
        self.device = device or torch.device("cpu")

    def rand(self):
        return float(torch.rand((), generator=self.gen, device=self.device))

    def normal(self, shape, mean=0.0, std=1.0):
        return torch.normal(
            mean=mean, std=std, size=shape, generator=self.gen, device=self.device
        )


# Individual transforms (PT)
class RandomFlipRotate(PairTransform):
    def __init__(self, p_h=0.5, p_v=0.5, max_deg=10.0, rng: Optional[RNG] = None):
        self.p_h, self.p_v, self.max_deg = p_h, p_v, max_deg
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor):
        image = _ensure_chw(image)
        mask = _ensure_chw(mask)
        if self.rng.rand() < self.p_h:
            image = torch.flip(image, [2])
            mask = torch.flip(mask, [2])
        if self.rng.rand() < self.p_v:
            image = torch.flip(image, [1])
            mask = torch.flip(mask, [1])
        angle = (self.rng.rand() * 2 - 1) * self.max_deg
        if abs(angle) > 1e-3:
            th = np.deg2rad(angle)
            c, s = np.cos(th), np.sin(th)
            A = torch.tensor(
                [[c, -s, 0.0], [s, c, 0.0]], dtype=torch.float32, device=image.device
            ).unsqueeze(0)
            H, W = image.shape[-2:]
            grid = F.affine_grid(A, size=(1, image.shape[0], H, W), align_corners=False)
            image = F.grid_sample(
                image.unsqueeze(0),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0)
            mask = F.grid_sample(
                mask.unsqueeze(0),
                grid,
                mode="nearest",
                padding_mode="border",
                align_corners=False,
            ).squeeze(0)
        return image, mask


class IntensityShift(PairTransform):
    def __init__(self, mul=(0.95, 1.05), add=(-0.02, 0.02), rng: Optional[RNG] = None):
        self.mul, self.add = mul, add
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor):
        m = self.mul[0] + self.rng.rand() * (self.mul[1] - self.mul[0])
        a = self.add[0] + self.rng.rand() * (self.add[1] - self.add[0])
        image = torch.clamp(image * m + a, 0, 1)
        return image, mask


class HUWindowing(PairTransform):
    def __init__(
        self,
        center=40.0,
        width=400.0,
        input_hu_range: Optional[Tuple[float, float]] = (0.0, 1.0),
    ):
        self.c = float(center)
        self.w = float(width)
        self.r = input_hu_range

    def __call__(self, image: Tensor, mask: Tensor):
        image = _ensure_chw(image).to(torch.float32)
        mask = _ensure_chw(mask)
        if self.r is not None:
            lo, hi = self.r
            image_hu = image * (hi - lo) + lo
        else:
            image_hu = image
        wmin, wmax = self.c - self.w / 2, self.c + self.w / 2
        image_hu = torch.clamp(image_hu, wmin, wmax)
        image = (image_hu - wmin) / max(self.w, 1e-6)
        return image, mask


class ElasticDeformation(PairTransform):
    def __init__(self, alpha=12.0, sigma=6.0, rng: Optional[RNG] = None):
        self.alpha = float(alpha)
        self.sigma = float(sigma)
        self.rng = rng or RNG()

    @staticmethod
    def _gauss2d(sigma, device):
        k = max(3, int(6 * sigma) | 1)
        ax = torch.arange(k, device=device) - k // 2
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        k2 = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        k2 = k2 / k2.sum().clamp_min(1e-8)
        return k2

    def __call__(self, image: Tensor, mask: Tensor):
        image = _ensure_chw(image)
        mask = _ensure_chw(mask)
        C, H, W = image.shape
        dev = image.device
        disp = self.rng.normal((2, H, W)).to(dev, torch.float32)
        k = self._gauss2d(self.sigma, dev).expand(2, 1, -1, -1)
        disp = F.conv2d(
            disp.unsqueeze(0), k, padding=k.shape[-1] // 2, groups=2
        ).squeeze(0)
        disp = disp * (self.alpha / max(H, W))
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=dev),
            torch.linspace(-1, 1, W, device=dev),
            indexing="ij",
        )
        grid = torch.stack((xx, yy), dim=-1)
        grid = grid + torch.stack((disp[1] * 2.0, disp[0] * 2.0), dim=-1)
        grid = grid.clamp(-1, 1)
        image = F.grid_sample(
            image.unsqueeze(0),
            grid.unsqueeze(0),
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid.unsqueeze(0),
            mode="nearest",
            padding_mode="border",
            align_corners=True,
        ).squeeze(0)
        return image, mask


class RandomScale(PairTransform):
    def __init__(self, scale=(0.9, 1.1), rng: Optional[RNG] = None):
        self.scale = scale
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor):
        s = self.scale[0] + self.rng.rand() * (self.scale[1] - self.scale[0])
        A = torch.tensor(
            [[s, 0, 0], [0, s, 0]], dtype=torch.float32, device=image.device
        ).unsqueeze(0)
        H, W = image.shape[-2:]
        grid = F.affine_grid(A, size=(1, image.shape[0], H, W), align_corners=False)
        image = F.grid_sample(
            image.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)
        mask = F.grid_sample(
            mask.unsqueeze(0),
            grid,
            mode="nearest",
            padding_mode="border",
            align_corners=False,
        ).squeeze(0)
        return image, mask


class GammaCorrection(PairTransform):
    def __init__(self, gamma=(0.7, 1.5), rng: Optional[RNG] = None):
        self.gamma = gamma
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor):
        g = self.gamma[0] + self.rng.rand() * (self.gamma[1] - self.gamma[0])
        image = image.clamp_min(1e-6).pow(g).clamp(0, 1)
        return image, mask


class GaussianNoise(PairTransform):
    def __init__(self, sigma=(0.0, 0.02), rng: Optional[RNG] = None):
        self.sigma = sigma
        self.rng = rng or RNG()

    def __call__(self, image: Tensor, mask: Tensor):
        s = self.sigma[0] + self.rng.rand() * (self.sigma[1] - self.sigma[0])
        noise = self.rng.normal(image.shape, std=s).to(image.device, image.dtype)
        image = (image + noise).clamp(0, 1)
        return image, mask


# Build PT presets (names compatible with your CLI)
def _build_pt_preset(name: str, seed: Optional[int]) -> PairTransform:
    rng = RNG(seed)
    key = name.lower()
    if key == "none":
        return ComposePT([])
    if key == "basic":
        return ComposePT(
            [
                RandomFlipRotate(max_deg=10.0, rng=rng),
                IntensityShift(mul=(0.95, 1.05), add=(-0.02, 0.02), rng=rng),
            ]
        )
    if key == "basic+elastic":
        return ComposePT(
            [
                RandomFlipRotate(max_deg=7.0, rng=rng),
                IntensityShift(mul=(0.95, 1.05), add=(-0.02, 0.02), rng=rng),
                ElasticDeformation(alpha=12.0, sigma=6.0, rng=rng),
            ]
        )
    if key == "hu+basic":
        return ComposePT(
            [
                HUWindowing(center=40.0, width=400.0, input_hu_range=(0.0, 1.0)),
                RandomFlipRotate(max_deg=10.0, rng=rng),
                IntensityShift(mul=(0.95, 1.05), add=(-0.02, 0.02), rng=rng),
            ]
        )
    if key == "dkfz-like":
        return ComposePT(
            [
                RandomFlipRotate(max_deg=10.0, rng=rng),
                RandomScale(scale=(0.9, 1.1), rng=rng),
                ElasticDeformation(alpha=12.0, sigma=6.0, rng=rng),
                IntensityShift(mul=(0.95, 1.05), add=(-0.02, 0.02), rng=rng),
                GammaCorrection(gamma=(0.7, 1.5), rng=rng),
                GaussianNoise(sigma=(0.0, 0.02), rng=rng),
            ]
        )
    raise KeyError(f"Unknown augmentation preset '{name}'")


class AugmentedDataset(torch.utils.data.Dataset):
    """
    Wraps SliceDataset to apply PT transforms during training.
    Ensures masks remain strict one-hot float in [K,H,W].
    """

    def __init__(
        self, base_ds: SliceDataset, transform: PairTransform, K_guess: int = 5
    ):
        self.base = base_ds
        self.transform = transform
        ex = self.base[0]
        m = ex["gts"]
        if isinstance(m, torch.Tensor) and m.ndim == 3 and m.shape[0] > 1:
            self.K = m.shape[0]
        else:
            self.K = K_guess

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        item = self.base[idx]
        image: Tensor = item["images"].to(torch.float32)  # [1,H,W]
        mask: Tensor = item["gts"]  # [K,H,W] or [1,H,W] labels

        # Normalize mask to one-hot float [K,H,W]
        if isinstance(mask, torch.Tensor):
            if mask.ndim == 3 and mask.shape[0] > 1:
                mask_oh = mask.to(torch.float32)
            else:
                labels = mask.to(torch.long)  # [1,H,W]
                mask_oh = class2one_hot(labels, K=self.K)[0].to(torch.float32)
        else:
            arr = np.asarray(mask)
            if arr.ndim == 3 and arr.shape[0] > 1:
                mask_oh = torch.from_numpy(arr.astype(np.float32))
            else:
                onehot = np.eye(self.K, dtype=np.float32)[arr.astype(np.int64)]
                mask_oh = torch.from_numpy(np.transpose(onehot, (2, 0, 1)))

        img_t, msk_t = self.transform(image, mask_oh)

        # Re-quantize mask to strict one-hot after any warp
        hard = msk_t.argmax(dim=0, keepdim=True).to(torch.long)  # [1,H,W]
        msk_t = _labels_to_onehot(hard, self.K)  # [K,H,W] float

        out = dict(item)
        out["images"] = torch.clamp(img_t, 0, 1)
        out["gts"] = msk_t
        return out


# =============================================================================
# Public API for main.py
# =============================================================================


def add_augmentation_cli(parser: argparse.ArgumentParser) -> None:
    grp = parser.add_argument_group("Augmentation")
    grp.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=["none", "basic", "basic+elastic", "hu+basic", "dkfz-like"],
        help="Training-time augmentation preset",
    )
    grp.add_argument(
        "--aug-seed", type=int, default=None, help="Deterministic augmentation seed"
    )


def wrap_train_dataset_if_needed(
    train_set: SliceDataset, args
) -> torch.utils.data.Dataset:
    aug_name = getattr(args, "aug", "none")
    if aug_name and aug_name != "none":
        tfm = _build_pt_preset(aug_name, seed=getattr(args, "aug_seed", None))
        return AugmentedDataset(train_set, tfm)
    return train_set


def save_aug_examples(
    dataset: torch.utils.data.Dataset,
    dest: Path,
    n_samples: int = 4,
    n_augments: int = 3,
) -> None:
    """
    Save a few (orig, aug##) pairs from the *training* dataset to inspect transforms.
    Works whether `dataset` is already AugmentedDataset or plain SliceDataset.
    """
    dest = Path(dest) / "aug_examples"
    dest.mkdir(parents=True, exist_ok=True)

    # If dataset is already augmented, extract its base and transform;
    # else, preview without transform (orig only).
    if isinstance(dataset, AugmentedDataset):
        base = dataset.base
        transform = dataset.transform
        K = dataset.K
    else:
        base = dataset
        transform = None
        # try to infer K
        ex = base[0]["gts"]
        K = (
            ex.shape[0]
            if isinstance(ex, torch.Tensor) and ex.ndim == 3 and ex.shape[0] > 1
            else 5
        )

    for idx in range(min(n_samples, len(base))):
        item = base[idx]
        img = item["images"].to(torch.float32)  # [1,H,W]
        gt = item["gts"]
        if isinstance(gt, torch.Tensor):
            if gt.ndim == 3 and gt.shape[0] > 1:
                m_oh = gt.to(torch.float32)
            else:
                m_oh = class2one_hot(gt.to(torch.long)[None, ...], K=K)[0].to(
                    torch.float32
                )
        else:
            arr = np.asarray(gt)
            if arr.ndim == 3 and arr.shape[0] > 1:
                m_oh = torch.from_numpy(arr.astype(np.float32))
            else:
                onehot = np.eye(K, dtype=np.float32)[arr.astype(np.int64)]
                m_oh = torch.from_numpy(np.transpose(onehot, (2, 0, 1)))

        # save original
        _save_png(img.detach().cpu().numpy(), dest / f"sample{idx:02d}_orig.png")

        # overlay orig mask for reference
        mask_rgb = _colorize_labels(m_oh.detach().cpu().numpy())
        rgb = np.repeat(img.detach().cpu().numpy(), 3, axis=0)
        overlay = (rgb * 0.6 + mask_rgb * 0.4).astype(np.float32)
        _save_png(overlay, dest / f"sample{idx:02d}_orig_overlay.png")

        # save a few augmentations
        if transform is not None:
            for j in range(n_augments):
                img_aug, msk_aug = transform(img, m_oh)
                # re-quantize to one-hot for clean visualization
                hard = msk_aug.argmax(dim=0, keepdim=True).to(torch.long)
                msk_aug = _labels_to_onehot(hard, K)

                _save_png(
                    img_aug.detach().cpu().numpy(),
                    dest / f"sample{idx:02d}_aug{j:02d}.png",
                )

                m_rgb = _colorize_labels(msk_aug.detach().cpu().numpy())
                rgbA = np.repeat(img_aug.detach().cpu().numpy(), 3, axis=0)
                overlayA = (rgbA * 0.6 + m_rgb * 0.4).astype(np.float32)
                _save_png(overlayA, dest / f"sample{idx:02d}_aug{j:02d}_overlay.png")
