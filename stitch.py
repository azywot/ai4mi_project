#!/usr/bin/env python3

import argparse
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import nibabel as nib
from collections import defaultdict


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stitch 2D slices back into 3D volumes")
    parser.add_argument("--data_folder", type=str, required=True,
                        help="Folder containing sliced data (e.g., data/prediction/best_epoch/val)")
    parser.add_argument("--dest_folder", type=str, required=True,
                        help="Destination folder for stitched predictions (e.g., val/pred)")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="Number of classes (e.g., 5)")
    parser.add_argument("--grp_regex", type=str, required=True,
                        help="Pattern for filename grouping "
                             "(e.g., '(Patient\\d+)_\\d{4}' or '(Patient_\\d\\d)_\\d\\d\\d\\d')")
    parser.add_argument("--source_scan_pattern", type=str, required=True,
                        help="Pattern to original GT scans for metadata "
                             "(e.g., 'data/segthor_fixed/train/{id_}/GT.nii.gz')")
    parser.add_argument("--copy-gt", action="store_true",
                        help="If set, copy the original GT files into val/gt/")
    return parser.parse_args()


def group_slices_by_patient(data_folder: Path, grp_regex: str) -> Dict[str, List[Path]]:
    """Group slice files by patient ID using the provided regex pattern."""
    pattern = re.compile(grp_regex)
    patient_slices = defaultdict(list)

    # Find all .png files in the data folder
    for slice_file in data_folder.glob("*.png"):
        match = pattern.search(slice_file.name)
        if match:
            patient_id = match.group(1)
            patient_slices[patient_id].append(slice_file)
        else:
            print(f"Warning: Could not extract patient ID from {slice_file.name}")

    # Sort slices by filename
    for patient_id in patient_slices:
        patient_slices[patient_id].sort(key=lambda x: x.name)

    return dict(patient_slices)


def load_slice_as_segmentation(slice_path: Path, num_classes: int) -> np.ndarray:
    """Load a slice image and convert it to segmentation format."""
    from PIL import Image

    img = Image.open(slice_path)
    slice_array = np.array(img)

    if slice_array.ndim == 2:
        return slice_array.astype(np.uint8)

    # take the first channel if multiple channels
    if slice_array.ndim == 3:
        return slice_array[:, :, 0].astype(np.uint8)

    return slice_array.astype(np.uint8)


def get_original_metadata(source_scan_pattern: str, patient_id: str) -> Tuple[Tuple[int, int, int], np.ndarray]:
    """Get the original scan metadata (shape and affine) from the source scan."""
    source_path = Path(source_scan_pattern.format(id_=patient_id))

    if not source_path.exists():
        raise FileNotFoundError(f"Source scan not found: {source_path}")

    # Load the original scan to get metadata
    original_nii = nib.load(str(source_path))
    original_shape = original_nii.shape  # (H, W, D)
    original_affine = original_nii.affine

    if len(original_shape) < 3:
        raise ValueError(f"Expected at least 3D NIfTI for {patient_id}, got shape {original_shape}")

    return (int(original_shape[0]), int(original_shape[1]), int(original_shape[2])), original_affine


def stitch_patient_slices(slice_files: List[Path], num_classes: int,
                          original_shape: Tuple[int, int, int]) -> np.ndarray:
    """Stitch 2D slices back into a 3D volume, matching the original in-plane size."""
    height, width, depth = original_shape  # (H, W, D)
    volume = np.zeros((height, width, depth), dtype=np.uint8)

    for i, slice_file in enumerate(slice_files):
        if i >= depth:
            print(f"Warning: More slices ({len(slice_files)}) than expected depth ({depth})")
            break

        slice_data = load_slice_as_segmentation(slice_file, num_classes)

        # Resize slice to (height, width) if needed
        if slice_data.shape != (height, width):
            # print(f"Resizing slice {slice_file.name} from {slice_data.shape} to {(height, width)}")
            from skimage.transform import resize
            slice_data = resize(
                slice_data, (height, width),
                mode='constant', preserve_range=True,
                anti_aliasing=False, order=0  # nearest-neighbor for label maps
            ).astype(np.uint8)

        volume[:, :, i] = slice_data

    if len(slice_files) < depth:
        print(f"Warning: Fewer slices ({len(slice_files)}) than expected depth ({depth})")

    return volume


def save_stitched_volume(volume: np.ndarray, affine: np.ndarray,
                         dest_path: Path, patient_id: str) -> None:
    """Save the stitched 3D volume as a NIfTI file."""
    nii_img = nib.Nifti1Image(volume, affine)

    dest_path.mkdir(parents=True, exist_ok=True)
    output_file = dest_path / f"{patient_id}.nii.gz"
    nib.save(nii_img, str(output_file))
    print(f"Saved stitched volume: {output_file}")


def copy_gt_files(patient_ids: List[str], source_scan_pattern: str, dest_folder: Path):
    """Copy original GT files to the destination folder for evaluation."""
    dest_folder.mkdir(parents=True, exist_ok=True)
    for pid in patient_ids:
        source_path = Path(source_scan_pattern.format(id_=pid))
        if not source_path.exists():
            print(f"Warning: GT not found for {pid} at {source_path}")
            continue

        dest_path = dest_folder / f"{pid}.nii.gz"
        shutil.copy(source_path, dest_path)
        print(f"Copied GT for {pid} to {dest_path}")


def main():
    args = get_args()

    data_folder = Path(args.data_folder)
    dest_folder = Path(args.dest_folder)

    if not data_folder.exists():
        raise FileNotFoundError(f"Data folder not found: {data_folder}")

    print(f"Stitching slices from: {data_folder}")
    print(f"Output destination: {dest_folder}")
    print(f"Number of classes: {args.num_classes}")
    print(f"Grouping regex: {args.grp_regex}")
    print(f"Source scan pattern: {args.source_scan_pattern}")
    print(f"Copy GT enabled: {args.copy_gt}")

    # Group slices by patient
    patient_slices = group_slices_by_patient(data_folder, args.grp_regex)

    if not patient_slices:
        raise ValueError("No patient slices found. Check your data folder and regex pattern.")

    print(f"Found {len(patient_slices)} patients")

    # Process each patient for predictions
    for patient_id, slice_files in patient_slices.items():
        print(f"\nProcessing {patient_id} with {len(slice_files)} slices...")

        try:
            # Get original scan metadata
            original_shape, original_affine = get_original_metadata(
                args.source_scan_pattern, patient_id
            )

            print(f"Original shape: {original_shape}")
            volume = stitch_patient_slices(slice_files, args.num_classes, original_shape)
            print(f"Stitched volume shape: {volume.shape}")
            save_stitched_volume(volume, original_affine, dest_folder, patient_id)

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue

    # copy GT files
    if args.copy_gt:
        gt_dest = dest_folder.parent / "gt"
        print(f"\nCopying GT files to {gt_dest}...")
        copy_gt_files(list(patient_slices.keys()), args.source_scan_pattern, gt_dest)

    print(f"\nStitching completed. Results saved to: {dest_folder}")
    if args.copy_gt:
        print(f"GT files copied to: {gt_dest}")


if __name__ == "__main__":
    main()
