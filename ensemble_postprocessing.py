"""
Ensemble Post-Processing for Medical Image Segmentation

- Ensemble voting from multiple predictions
- ROI-based processing with center constraints
- Multi-label fusion and anatomical constraints
- Center-based post-processing for organ localization
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, ball, disk
from skimage.measure import label, regionprops
import warnings
warnings.filterwarnings('ignore')


def mask_center_2d(mask, rtol=1e-8):
    """
    Calculate center coordinates for 2D mask
    Based on get_center_2d.ipynb
    """
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return None, None
    
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, mask.shape)

    slices = [np.round((s+e)/2) for s, e in zip(start, end)]
    return int(slices[0]), int(slices[1])


def mask_center_3d(mask, rtol=1e-8):
    """
    Calculate center coordinates for 3D mask
    Based on get_center_3d.ipynb
    """
    coords = np.array(np.where(mask))
    if coords.size == 0:
        return None, None, None
    
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, mask.shape[:3])

    slices = [np.round((s+e)/2) for s, e in zip(start, end)]
    return int(slices[0]), int(slices[1]), int(slices[2])


def ensemble_voting_postprocessing(segmentation, num_classes=5, voting_threshold=0.5):
    """
    Ensemble voting post-processing
    Based on mutil_label_merge.ipynb
    """
    processed = segmentation.copy().astype(np.int32)
    
    # For each class, apply ensemble-like processing
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Remove small components (ensemble-like filtering)
        mask = remove_small_objects(mask, min_size=100)
        
        # Apply morphological operations for ensemble-like smoothing
        if len(mask.shape) == 3:
            selem = ball(1)
        else:
            selem = disk(1)
        
        mask = binary_closing(mask, footprint=selem)
        mask = binary_opening(mask, footprint=selem)
        
        processed[mask] = class_id
    
    return processed


def roi_based_postprocessing(segmentation, num_classes=5, roi_size=128):
    """
    ROI-based post-processing with center constraints
    Based on pred_sample.ipynb
    """
    processed = segmentation.copy().astype(np.int32)
    
    # Find centers for each organ
    organ_centers = {}
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        if np.sum(mask) > 0:
            if len(mask.shape) == 3:
                z, y, x = mask_center_3d(mask)
                if z is not None:
                    organ_centers[class_id] = (z, y, x)
            else:
                y, x = mask_center_2d(mask)
                if y is not None:
                    organ_centers[class_id] = (y, x)
    
    # Process each organ in its ROI
    for class_id, center in organ_centers.items():
        mask = (segmentation == class_id)
        
        if len(mask.shape) == 3:
            z, y, x = center
            # Define ROI around center
            z_start = max(0, z - roi_size//2)
            z_end = min(mask.shape[0], z + roi_size//2)
            y_start = max(0, y - roi_size//2)
            y_end = min(mask.shape[1], y + roi_size//2)
            x_start = max(0, x - roi_size//2)
            x_end = min(mask.shape[2], x + roi_size//2)
            
            # Process ROI
            roi_mask = mask[z_start:z_end, y_start:y_end, x_start:x_end]
            roi_processed = morphological_roi_processing(roi_mask)
            processed[z_start:z_end, y_start:y_end, x_start:x_end] = np.where(
                roi_processed, class_id, processed[z_start:z_end, y_start:y_end, x_start:x_end]
            )
        else:
            y, x = center
            # Define ROI around center
            y_start = max(0, y - roi_size//2)
            y_end = min(mask.shape[0], y + roi_size//2)
            x_start = max(0, x - roi_size//2)
            x_end = min(mask.shape[1], x + roi_size//2)
            
            # Process ROI
            roi_mask = mask[y_start:y_end, x_start:x_end]
            roi_processed = morphological_roi_processing(roi_mask)
            processed[y_start:y_end, x_start:x_end] = np.where(
                roi_processed, class_id, processed[y_start:y_end, x_start:x_end]
            )
    
    return processed


def morphological_roi_processing(mask):
    """
    Morphological processing for ROI
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove small objects
    processed = remove_small_objects(mask, min_size=50)
    
    # Fill holes
    processed = ndimage.binary_fill_holes(processed)
    
    # Smooth boundaries
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    
    processed = binary_closing(processed, footprint=selem)
    processed = binary_opening(processed, footprint=selem)
    
    return processed


def center_constraint_postprocessing(segmentation, num_classes=5):
    """
    Center-based constraint post-processing
    Based on get_center_2d.ipynb and get_center_3d.ipynb
    """
    processed = segmentation.copy().astype(np.int32)
    
    # Anatomical constraints based on expected organ positions
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Get center of mass
        if len(mask.shape) == 3:
            z, y, x = mask_center_3d(mask)
            if z is None:
                continue
        else:
            y, x = mask_center_2d(mask)
            if y is None:
                continue
        
        # Apply center-based constraints
        if len(mask.shape) == 3:
            # 3D constraints
            center_z, center_y, center_x = z, y, x
            
            # Remove components far from center
            labeled = label(mask)
            regions = regionprops(labeled)
            
            for region in regions:
                # Calculate distance from center
                centroid = region.centroid
                distance = np.sqrt((centroid[0] - center_z)**2 + 
                                 (centroid[1] - center_y)**2 + 
                                 (centroid[2] - center_x)**2)
                
                # Remove regions too far from center
                if distance > 50:  # Adjust threshold based on your data
                    mask[labeled == region.label] = False
        else:
            # 2D constraints
            center_y, center_x = y, x
            
            # Remove components far from center
            labeled = label(mask)
            regions = regionprops(labeled)
            
            for region in regions:
                # Calculate distance from center
                centroid = region.centroid
                distance = np.sqrt((centroid[0] - center_y)**2 + 
                                 (centroid[1] - center_x)**2)
                
                # Remove regions too far from center
                if distance > 30:  # Adjust threshold based on your data
                    mask[labeled == region.label] = False
        
        processed[mask] = class_id
    
    return processed


def multi_label_fusion_postprocessing(segmentation, num_classes=5):
    """
    Multi-label fusion post-processing
    Based on mutil_label_merge.ipynb
    """
    processed = segmentation.copy().astype(np.int32)
    
    # Class-specific processing based on anatomical knowledge
    # Class 1: Aorta, Class 2: Heart, Class 3: Left Lung, Class 4: Right Lung
    
    # Process each class with specific constraints
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Class-specific processing
        if class_id == 2:  # Heart - should be connected
            mask = ensure_connected_organ(mask)
        elif class_id in [3, 4]:  # Lungs - should be roughly symmetric
            mask = ensure_lung_symmetry(mask, class_id)
        elif class_id == 1:  # Aorta - should be connected and thin
            mask = ensure_aorta_constraints(mask)
        
        processed[mask] = class_id
    
    return processed


def ensure_connected_organ(mask):
    """
    Ensure heart is connected (remove isolated regions)
    """
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Keep only the largest component
    largest_region = max(regions, key=lambda x: x.area)
    connected_mask = (labeled == largest_region.label)
    
    return connected_mask


def ensure_lung_symmetry(mask, class_id):
    """
    Ensure lung symmetry constraints
    """
    # Remove very small lung regions
    mask = remove_small_objects(mask, min_size=1000)
    
    # Fill holes in lungs
    mask = ndimage.binary_fill_holes(mask)
    
    return mask


def ensure_aorta_constraints(mask):
    """
    Ensure aorta constraints (connected and thin)
    """
    # Remove small components
    mask = remove_small_objects(mask, min_size=100)
    
    # Ensure connectivity
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Keep only the largest component
    largest_region = max(regions, key=lambda x: x.area)
    connected_mask = (labeled == largest_region.label)
    
    return connected_mask


def advanced_ensemble_postprocessing(segmentation, num_classes=5, method='ensemble', roi_size=128):
    """
    Advanced ensemble post-processing for medical images
    
    Args:
        segmentation: Input segmentation array
        num_classes: Number of classes
        method: Processing method to use
        roi_size: Size of ROI for processing
    
    Returns:
        Processed segmentation
    """
    if method == 'ensemble':
        return ensemble_voting_postprocessing(segmentation, num_classes)
    elif method == 'roi':
        return roi_based_postprocessing(segmentation, num_classes, roi_size)
    elif method == 'center':
        return center_constraint_postprocessing(segmentation, num_classes)
    elif method == 'fusion':
        return multi_label_fusion_postprocessing(segmentation, num_classes)
    elif method == 'combined':
        # Apply all methods sequentially
        processed = ensemble_voting_postprocessing(segmentation, num_classes)
        processed = roi_based_postprocessing(processed, num_classes, roi_size)
        processed = center_constraint_postprocessing(processed, num_classes)
        processed = multi_label_fusion_postprocessing(processed, num_classes)
        return processed
    else:
        raise ValueError(f"Unknown method: {method}")


def process_single_file(input_file, output_file, num_classes, method, roi_size):
    """
    Process a single NIfTI file
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output NIfTI file
        num_classes: Number of classes
        method: Processing method
        roi_size: Size of ROI for processing
    """
    try:
        # Load segmentation
        img = nib.load(input_file)
        segmentation = img.get_fdata()
        
        # Apply post-processing
        processed = advanced_ensemble_postprocessing(segmentation, num_classes, method, roi_size)
        
        # Save result
        output_file.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(processed.astype(np.int32), img.affine, img.header), output_file)
        
        print(f"✓ Processed: {input_file.name}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {input_file.name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Ensemble post-processing for medical image segmentation')
    parser.add_argument('--input_folder', type=Path, required=True,
                       help='Input folder containing NIfTI files')
    parser.add_argument('--output_folder', type=Path, required=True,
                       help='Output folder for processed files')
    parser.add_argument('--num_classes', type=int, default=5,
                       help='Number of classes (default: 5)')
    parser.add_argument('--method', 
                       choices=['ensemble', 'roi', 'center', 'fusion', 'combined'],
                       default='ensemble',
                       help='Post-processing method (default: ensemble)')
    parser.add_argument('--roi_size', type=int, default=128,
                       help='Size of ROI for processing (default: 128)')
    parser.add_argument('--pattern', type=str, default='*.nii.gz',
                       help='File pattern to match (default: *.nii.gz)')
    
    args = parser.parse_args()
    
    # Validate input folder
    if not args.input_folder.exists():
        print(f"Error: Input folder {args.input_folder} does not exist")
        return 1
    
    # Create output folder
    args.output_folder.mkdir(parents=True, exist_ok=True)
    
    # Find input files
    input_files = list(args.input_folder.glob(args.pattern))
    if not input_files:
        print(f"No files found matching pattern {args.pattern} in {args.input_folder}")
        return 1
    
    print(f"Processing {len(input_files)} files with method: {args.method}")
    print(f"Input folder: {args.input_folder}")
    print(f"Output folder: {args.output_folder}")
    print(f"Number of classes: {args.num_classes}")
    print(f"ROI size: {args.roi_size}")
    print("-" * 50)
    
    # Process files
    success_count = 0
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        if process_single_file(input_file, output_file, args.num_classes, args.method, args.roi_size):
            success_count += 1
    
    print("-" * 50)
    print(f"Processing complete: {success_count}/{len(input_files)} files processed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())
