#!/usr/bin/env python3
from __future__ import annotations

# augment_data.py — unified training + preview augmentations

import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
from PIL import Image

# -------------------------
# batchgenerators
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

from dataset import SliceDataset
from utils import class2one_hot

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


def _labels_to_onehot(labels: np.ndarray, K: int) -> np.ndarray:
    """Convert label array to one-hot encoding"""
    if labels.ndim == 3 and labels.shape[0] > 1:
        return labels  # already one-hot
    if labels.ndim == 3:
        labels = labels[0]  # [H,W]
    onehot = np.eye(K, dtype=np.float32)[labels.astype(np.int64)]
    return np.transpose(onehot, (2, 0, 1))  # [K,H,W]


# =============================================================================
# Batchgenerators augmentation pipeline
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
                multiplier_range=(0.5, 1.5), per_channel=True, p_per_sample=p_per_sample
            )
        )
    if gamma:
        tr.append(
            GammaTransform(
                gamma_range=(0.8, 1.3),
                invert_image=False,
                per_channel=True,
                retain_stats=True,
                p_per_sample=p_per_sample,
            )
        )
    if gaussian_noise:
        tr.append(
            GaussianNoiseTransform(
                noise_variance=(0.0, 0.03), p_per_sample=p_per_sample
            )
        )
    if rotations or scaling or elastic:
        tr.append(
            SpatialTransform(
                patch_size=list(patch_shape),
                patch_center_dist_from_border=[ps // 2 for ps in patch_shape],
                do_elastic_deform=elastic,
                alpha=(0.0, 600.0),
                sigma=(9.0, 10.0),
                do_rotation=rotations,
                angle_x=None,
                angle_y=None,
                angle_z=(-15.0 / 360 * 2 * np.pi, 15.0 / 360 * 2 * np.pi),
                do_scale=scaling,
                scale=(0.9, 1.1),
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
    elif name == "basic+elastic":
        return build_bg_compose(
            mirror=True,
            rotations=True,
            scaling=True,
            elastic=True,
            brightness=True,
            contrast=False,
            gamma=False,
            gaussian_noise=False,
            p_per_sample=0.15,
            patch_shape=patch_shape,
        )
    elif name == "hu+basic":
        # HU windowing would be applied in preprocessing, so same as basic here
        return build_bg_compose(
            mirror=True,
            rotations=True,
            scaling=False,
            elastic=False,
            brightness=True,
            contrast=True,
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
# Wrapper for training with batchgenerators
# =============================================================================


class AugmentedDataset(torch.utils.data.Dataset):
    """Wraps a dataset and applies batchgenerators augmentation on-the-fly"""

    def __init__(self, base_dataset: SliceDataset, compose: BGCompose, K: int = 5):
        self.base = base_dataset
        self.compose = compose
        self.K = K

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        item = self.base[idx]
        img = _to_chw_numpy(item["images"])  # [1,H,W]
        gt = item["gts"]

        # Convert to one-hot numpy if needed
        if isinstance(gt, torch.Tensor):
            if gt.ndim == 3 and gt.shape[0] > 1:
                seg = gt.detach().cpu().to(torch.float32).numpy()
            else:
                lbl = gt.detach().cpu().to(torch.long).numpy()
                seg = _labels_to_onehot(lbl, self.K)
        else:
            gt = np.asarray(gt)
            seg = _labels_to_onehot(gt, self.K)

        # Apply batchgenerators augmentation
        if len(self.compose.transforms) > 0:
            # Add batch dimension
            batch_dict = {
                "data": img[None, ...],  # [1,1,H,W]
                "seg": seg[None, ...],  # [1,K,H,W]
            }

            # Apply transforms
            for transform in self.compose.transforms:
                batch_dict = transform(**batch_dict)

            # Remove batch dimension
            img = batch_dict["data"][0]
            seg = batch_dict["seg"][0]

        # Convert back to torch tensors
        img_t = torch.from_numpy(img).float()
        seg_t = torch.from_numpy(seg).float()

        return {"images": img_t, "gts": seg_t}


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
    train_set: SliceDataset, args, patch_shape: Optional[Tuple[int, int]] = None
) -> torch.utils.data.Dataset:
    """Wrap dataset with batchgenerators augmentation if specified"""
    aug_name = getattr(args, "aug", "none")

    if aug_name and aug_name != "none":
        # Set seed if specified
        aug_seed = getattr(args, "aug_seed", None)
        if aug_seed is not None:
            np.random.seed(aug_seed)

        # Get patch shape from dataset if not provided
        if patch_shape is None:
            first = train_set[0]
            img = first["images"]
            if isinstance(img, torch.Tensor):
                H, W = img.shape[-2:]
            else:
                img = np.asarray(img)
                H, W = img.shape[-2:]
            patch_shape = (H, W)

        # Build compose
        compose = preset_compose(aug_name, patch_shape)

        # Wrap dataset
        return AugmentedDataset(train_set, compose, K=5)

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

    # If dataset is already augmented, extract its base and compose;
    # else, preview without transform (orig only).
    if isinstance(dataset, AugmentedDataset):
        base = dataset.base
        compose = dataset.compose
        K = dataset.K
    else:
        base = dataset
        compose = None
        # try to infer K
        ex = base[0]["gts"]
        K = (
            ex.shape[0]
            if isinstance(ex, torch.Tensor) and ex.ndim == 3 and ex.shape[0] > 1
            else 5
        )

    for idx in range(min(n_samples, len(base))):
        item = base[idx]
        img = _to_chw_numpy(item["images"])  # [1,H,W]
        gt = item["gts"]

        # Convert to one-hot
        if isinstance(gt, torch.Tensor):
            if gt.ndim == 3 and gt.shape[0] > 1:
                m_oh = gt.detach().cpu().to(torch.float32).numpy()
            else:
                lbl = gt.detach().cpu().to(torch.long).numpy()
                m_oh = _labels_to_onehot(lbl, K)
        else:
            gt = np.asarray(gt)
            m_oh = _labels_to_onehot(gt, K)

        # save original
        _save_png(img, dest / f"sample{idx:02d}_orig.png")

        # overlay orig mask for reference
        mask_rgb = _colorize_labels(m_oh)
        rgb = np.repeat(img, 3, axis=0)
        overlay = (rgb * 0.6 + mask_rgb * 0.4).astype(np.float32)
        _save_png(overlay, dest / f"sample{idx:02d}_orig_overlay.png")

        # save a few augmentations
        if compose is not None and len(compose.transforms) > 0:
            for j in range(n_augments):
                # Apply augmentation
                batch_dict = {
                    "data": img[None, ...],  # [1,1,H,W]
                    "seg": m_oh[None, ...],  # [1,K,H,W]
                }

                for transform in compose.transforms:
                    batch_dict = transform(**batch_dict)

                img_aug = batch_dict["data"][0]
                msk_aug = batch_dict["seg"][0]

                # re-quantize to one-hot for clean visualization
                labels = np.argmax(msk_aug, axis=0)
                msk_aug = _labels_to_onehot(labels, K)

                _save_png(img_aug, dest / f"sample{idx:02d}_aug{j:02d}.png")

                m_rgb = _colorize_labels(msk_aug)
                rgbA = np.repeat(img_aug, 3, axis=0)
                overlayA = (rgbA * 0.6 + m_rgb * 0.4).astype(np.float32)
                _save_png(overlayA, dest / f"sample{idx:02d}_aug{j:02d}_overlay.png")
