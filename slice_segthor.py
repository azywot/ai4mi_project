import pickle
import random
import argparse
import warnings
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from typing import Callable

import numpy as np
import nibabel as nib
from skimage.transform import resize
from PIL import Image

from utils import map_, tqdm_


"""
TODO: Implement image normalisation.
CT images have a wide range of intensity values (Hounsfield units)
Goal: normalize an image array to the range [0, 255]  and return it as a dtype=uint8
Which is compatible with standard image formats (PNG)
"""
def norm_arr(img: np.ndarray) -> np.ndarray:
    # TODO: your code here
    # The case where all pixels have the same value
    min_val, max_val = np.min(img), np.max(img)
    if min_val == max_val:
        return np.zeros_like(img, dtype=np.uint8)

    # Min-max normalization to scale values to the [0, 1]
    normalized_img = (img - min_val) / (max_val - min_val)

    # Scale to [0, 255] and convert to uint8
    scaled_img = (normalized_img * 255).astype(np.uint8)

    return scaled_img


def sanity_ct(ct, x, y, z, dx, dy, dz) -> bool:
    assert ct.dtype in [np.int16, np.int32], ct.dtype
    assert -1000 <= ct.min(), ct.min()
    assert ct.max() <= 31743, ct.max()

    assert 0.896 <= dx <= 1.37, dx  # Rounding error
    assert dx == dy
    assert 2 <= dz <= 3.7, dz

    assert (x, y) == (512, 512)
    assert x == y
    assert 135 <= z <= 284, z

    return True

def sanity_gt(gt, ct) -> bool:
    assert gt.shape == ct.shape
    assert gt.dtype in [np.uint8], gt.dtype

    # Do the test on 3d: assume all organs are present..
    assert set(np.unique(gt)) == set(range(5))

    return True


"""
TODO: Implement patient slicing.
Context:
  - Given an ID and paths, load the NIfTI CT volume and (if not test_mode) the GT volume.
  - Validate with sanity_ct / sanity_gt.
  - Normalise CT with norm_arr().
  - Slice the 3D volumes into 2D slices, resize to `shape`, and save PNGs.
  - Currently we have groundtruth masks marked as {0,1,2,3,4} but those values are hard to distinguish in a grayscale png.
    Multiplying by 63 maps them to {0,63,126,189,252}, which keeps labels visually distinct in a grayscale PNG.
    You can use the following code, which works for already sliced 2d images:
    gt_slice *= 63
    assert gt_slice.dtype == np.uint8, gt_slice.dtype
    assert set(np.unique(gt_slice)) <= set([0, 63, 126, 189, 252]), np.unique(gt_slice)
  - Return the original voxel spacings (dx, dy, dz).

Hints:
  - Use nibabel to load NIfTI images.
  - Use skimage.transform.resize (tip: anti_aliasing might be useful)
  - The PNG files should be stored in the dest_path, organised into separate subfolders: train/img, train/gt, val/img, and val/gt
  - Use consistent filenames: e.g. f"{id_}_{idz:04d}.png" inside subfolders "img" and "gt"; where idz is the slice index.
"""

def slice_patient(id_: str, dest_path: Path, source_path: Path, shape: tuple[int, int], test_mode=False)\
        -> tuple[float, float, float]:

    id_path: Path = source_path / ("train" if not test_mode else "test") / id_
    ct_path: Path = (id_path / f"{id_}.nii.gz")
    assert id_path.exists()
    assert ct_path.exists()

    # --------- FILL FROM HERE -----------
    # Load the NIfTI file for the CT scan
    ct_nii = nib.load(ct_path)
    ct = ct_nii.get_fdata()

    # Extract voxel spacing and dimensions
    dx, dy, dz = ct_nii.header.get_zooms()
    x, y, z = ct.shape

    # Convert CT data to int
    ct = ct.astype(np.int16)
    sanity_ct(ct, x, y, z, dx, dy, dz)

    # Load gt data if not in test mode
    if not test_mode:
        gt_path: Path = id_path / f"GT.nii.gz"
        assert gt_path.exists()
        gt_nii = nib.load(gt_path)
        gt = gt_nii.get_fdata().astype(np.uint8)

        sanity_gt(gt, ct)

    # Normalize the CT
    ct_normalized = norm_arr(ct)

    # Create destination paths
    img_dest = dest_path / "img"
    img_dest.mkdir(parents=True, exist_ok=True)
    if not test_mode:
        gt_dest = dest_path / "gt"
        gt_dest.mkdir(parents=True, exist_ok=True)

    # Slice the volume along the z-axis
    print(f"Slicing patient {id_} with shape {ct.shape} and spacing {(dx, dy, dz)}")
    for idz in range(z):
        ct_slice = ct_normalized[:, :, idz]

        # Resize the CT slice with bi-linear interpolation and anti-aliasing
        ct_resized = resize(ct_slice,
                              shape,
                              order=1,
                              preserve_range=True,
                              anti_aliasing=True).astype(np.uint8)

        filename = f"{id_}_{idz:04d}.png"
        Image.fromarray(ct_resized).save(img_dest / filename)

        if not test_mode:
            gt_slice = gt[:, :, idz]
            # Resize GT slice with NN-interpolation to preserve labels
            gt_resized = resize(gt_slice,
                                shape,
                                order=0,
                                preserve_range=True,
                                anti_aliasing=False).astype(np.uint8)

            # Map labels to be visually distinct
            gt_resized *= 63
            assert gt_resized.dtype == np.uint8
            assert set(np.unique(gt_resized)) <= set([0, 63, 126, 189, 252])
            Image.fromarray(gt_resized).save(gt_dest / filename)

    return (dx, dy, dz)


"""
TODO: Implement a simple train/val split.
Requirements:
  - List patient IDs from <src_path>/train (folder names).
  - Shuffle them (respect a seed set in main()).
  - Take the first `retains` as validation, and the rest as training.
  - Return (training_ids, validation_ids).
"""

def get_splits(src_path: Path, retains: int) -> tuple[list[str], list[str]]:
    # TODO: your code here
    train_path = src_path / "train"
    assert train_path.exists(), f"Train path {train_path} does not exist"

    patient_ids = [folder.name for folder in train_path.iterdir() if folder.is_dir()]
    random.shuffle(patient_ids)

    validation_ids = patient_ids[:retains]
    training_ids = patient_ids[retains:]
    
    return training_ids, validation_ids

def main(args: argparse.Namespace):
    src_path: Path = Path(args.source_dir)
    dest_path: Path = Path(args.dest_dir)
    if not dest_path.exists():
        dest_path.mkdir(parents=True, exist_ok=True)

    assert src_path.exists()
    assert dest_path.exists()

    training_ids: list[str]
    validation_ids: list[str]
    training_ids, validation_ids = get_splits(src_path, args.retains)


    resolution_dict: dict[str, tuple[float, float, float]] = {}

    for mode, split_ids in zip(["train", "val"], [training_ids, validation_ids]):
        dest_mode: Path = dest_path / mode
        print(f"Slicing {len(split_ids)} pairs to {dest_mode}")

        pfun: Callable = partial(slice_patient,
                                 dest_path=dest_mode,
                                 source_path=src_path,
                                 shape=tuple(args.shape))

        resolutions: list[tuple[float, float, float]]
        iterator = tqdm_(split_ids)
        resolutions = list(map(pfun, iterator))

        for key, val in zip(split_ids, resolutions):
            resolution_dict[key] = val

    with open(dest_path / "spacing.pkl", 'wb') as f:
        pickle.dump(resolution_dict, f, pickle.HIGHEST_PROTOCOL)
        print(f"Saved spacing dictionnary to {f}")




def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description = "Slicing parameters")

    parser.add_argument('--source_dir', type=str, required=True)
    parser.add_argument('--dest_dir', type=str, required=True)
    parser.add_argument('--shape', type=int, nargs="+", default=[256, 256])
    parser.add_argument('--retains', type=int, default=10, help="Number of retained patient for the validation data")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    random.seed(args.seed)
    print(args)

    return args

if __name__ == "__main__":
    main(get_args())