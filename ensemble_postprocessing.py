"""
Ensemble Post-Processing for Medical Image Segmentation

- Ensemble voting from multiple predictions
- ROI-based processing with center constraints
- Multi-label fusion and anatomical constraints
- Center-based post-processing for organ localization
- Multiple Component Analysis (MCA) for artifact reduction
"""

import argparse
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import ndimage
from skimage.morphology import binary_closing, binary_opening, remove_small_objects, ball, disk
from skimage.measure import label, regionprops
from skimage.filters import gaussian
from scipy.spatial.distance import cdist
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


def multiple_component_analysis_postprocessing(segmentation, num_classes=5, 
                                             min_component_size=100, 
                                             max_components_per_class=3,
                                             artifact_threshold=0.3,
                                             connectivity_threshold=0.7):
    """
    Multiple Component Analysis (MCA) post-processing for artifact reduction
    
    This method analyzes components within each class and filters out artifacts
    based on size, shape, connectivity, and spatial relationships.
    
    Args:
        segmentation: Input segmentation array
        num_classes: Number of classes
        min_component_size: Minimum size for valid components
        max_components_per_class: Maximum number of components to keep per class
        artifact_threshold: Threshold for artifact detection (0-1)
        connectivity_threshold: Threshold for component connectivity analysis
    
    Returns:
        Processed segmentation with artifacts removed
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Apply MCA-based artifact reduction
        cleaned_mask = mca_artifact_reduction(
            mask, 
            min_component_size=min_component_size,
            max_components=max_components_per_class,
            artifact_threshold=artifact_threshold,
            connectivity_threshold=connectivity_threshold
        )
        
        processed[processed == class_id] = 0  # Clear original class
        processed[cleaned_mask] = class_id    # Set cleaned class
    
    return processed


def mca_artifact_reduction(mask, min_component_size=100, max_components=3,
                          artifact_threshold=0.3, connectivity_threshold=0.7):
    """
    Multiple Component Analysis for artifact reduction
    
    Args:
        mask: Binary mask for a single class
        min_component_size: Minimum size for valid components
        max_components: Maximum number of components to keep
        artifact_threshold: Threshold for artifact detection
        connectivity_threshold: Threshold for connectivity analysis
    
    Returns:
        Cleaned binary mask with artifacts removed
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Label connected components
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Step 2: Analyze component properties
    component_scores = []
    valid_components = []
    
    for region in regions:
        # Calculate component features
        features = analyze_component_features(region, mask.shape)
        
        # Calculate artifact score (lower is better)
        artifact_score = calculate_artifact_score(features, artifact_threshold)
        
        # Check if component meets minimum size requirement
        if region.area >= min_component_size:
            component_scores.append((region.label, artifact_score, features))
            valid_components.append(region.label)
    
    if not valid_components:
        return np.zeros_like(mask, dtype=bool)
    
    # Step 3: Sort components by artifact score (ascending - less artifacts first)
    component_scores.sort(key=lambda x: x[1])
    
    # Step 4: Select best components
    selected_components = []
    for i, (label, score, features) in enumerate(component_scores):
        if i >= max_components:
            break
        
        # Additional connectivity check for selected components
        if features['connectivity'] >= connectivity_threshold:
            selected_components.append(label)
    
    # Step 5: Create cleaned mask
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    for component_label in selected_components:
        cleaned_mask[labeled == component_label] = True
    
    # Step 6: Apply final morphological cleaning
    cleaned_mask = final_morphological_cleaning(cleaned_mask)
    
    return cleaned_mask


def analyze_component_features(region, image_shape):
    """
    Analyze features of a connected component for artifact detection
    
    Args:
        region: RegionProps object from skimage
        image_shape: Shape of the original image
    
    Returns:
        Dictionary of component features
    """
    features = {}
    
    # Basic geometric features
    features['area'] = region.area
    features['perimeter'] = region.perimeter
    features['eccentricity'] = region.eccentricity
    features['solidity'] = region.solidity
    features['extent'] = region.extent
    
    # Shape features
    features['compactness'] = (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else float('inf')
    features['aspect_ratio'] = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else float('inf')
    
    # Spatial features
    features['centroid_distance'] = calculate_centroid_distance(region, image_shape)
    features['boundary_ratio'] = calculate_boundary_ratio(region, image_shape)
    
    # Connectivity features
    features['connectivity'] = calculate_connectivity_score(region)
    
    # Density features
    features['density'] = region.area / region.bbox_area if region.bbox_area > 0 else 0
    
    return features


def calculate_artifact_score(features, threshold=0.3):
    """
    Calculate artifact score for a component (0-1, lower is better)
    
    Args:
        features: Dictionary of component features
        threshold: Threshold for artifact detection
    
    Returns:
        Artifact score (0-1)
    """
    score = 0.0
    
    # Shape-based artifact indicators
    if features['eccentricity'] > 0.9:  # Very elongated
        score += 0.2
    if features['solidity'] < 0.5:  # Very irregular
        score += 0.2
    if features['compactness'] > 2.0:  # Very non-compact
        score += 0.15
    if features['aspect_ratio'] > 5.0:  # Very elongated
        score += 0.15
    
    # Size-based artifact indicators
    if features['area'] < 50:  # Very small
        score += 0.3
    if features['density'] < 0.3:  # Very sparse
        score += 0.1
    
    # Spatial-based artifact indicators
    if features['centroid_distance'] > 0.8:  # Far from center
        score += 0.1
    if features['boundary_ratio'] > 0.7:  # Too close to boundary
        score += 0.1
    
    # Connectivity-based artifact indicators
    if features['connectivity'] < 0.5:  # Poor connectivity
        score += 0.2
    
    return min(score, 1.0)


def calculate_centroid_distance(region, image_shape):
    """
    Calculate normalized distance from component centroid to image center
    """
    image_center = np.array(image_shape) / 2
    centroid = np.array(region.centroid)
    
    # Normalize by image diagonal
    max_distance = np.linalg.norm(image_shape)
    distance = np.linalg.norm(centroid - image_center)
    
    return distance / max_distance if max_distance > 0 else 0


def calculate_boundary_ratio(region, image_shape):
    """
    Calculate ratio of component boundary touching image edges
    """
    bbox = region.bbox
    boundary_touches = 0
    
    # Check if component touches any image boundary
    for i, (start, end, size) in enumerate(zip(bbox[::2], bbox[1::2], image_shape)):
        if start <= 1 or end >= size - 1:
            boundary_touches += 1
    
    return boundary_touches / len(image_shape)


def calculate_connectivity_score(region):
    """
    Calculate connectivity score based on component properties
    """
    # Higher solidity and lower eccentricity indicate better connectivity
    connectivity = (region.solidity + (1 - region.eccentricity)) / 2
    return connectivity


def final_morphological_cleaning(mask):
    """
    Apply final morphological operations for cleaning
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove very small objects
    cleaned = remove_small_objects(mask, min_size=20)
    
    # Fill small holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Apply gentle smoothing
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    
    # Gentle closing to connect nearby regions
    cleaned = binary_closing(cleaned, footprint=selem)
    
    # Gentle opening to remove small protrusions
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def advanced_ensemble_postprocessing(segmentation, num_classes=5, method='ensemble', roi_size=128,
                                   min_component_size=100, max_components_per_class=3,
                                   artifact_threshold=0.3, connectivity_threshold=0.7):
    """
    Advanced ensemble post-processing for medical images
    
    Args:
        segmentation: Input segmentation array
        num_classes: Number of classes
        method: Processing method to use
        roi_size: Size of ROI for processing
        min_component_size: Minimum size for valid components (MCA method)
        max_components_per_class: Maximum number of components per class (MCA method)
        artifact_threshold: Threshold for artifact detection (MCA method)
        connectivity_threshold: Threshold for connectivity analysis (MCA method)
    
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
    elif method == 'mca':
        return multiple_component_analysis_postprocessing(
            segmentation, num_classes, min_component_size, 
            max_components_per_class, artifact_threshold, connectivity_threshold
        )
    elif method == 'combined':
        # Apply all methods sequentially
        processed = ensemble_voting_postprocessing(segmentation, num_classes)
        processed = roi_based_postprocessing(processed, num_classes, roi_size)
        processed = center_constraint_postprocessing(processed, num_classes)
        processed = multi_label_fusion_postprocessing(processed, num_classes)
        return processed
    elif method == 'mca_combined':
        # Apply MCA first, then other methods
        processed = multiple_component_analysis_postprocessing(
            segmentation, num_classes, min_component_size, 
            max_components_per_class, artifact_threshold, connectivity_threshold
        )
        processed = ensemble_voting_postprocessing(processed, num_classes)
        processed = roi_based_postprocessing(processed, num_classes, roi_size)
        processed = center_constraint_postprocessing(processed, num_classes)
        return processed
    else:
        raise ValueError(f"Unknown method: {method}")


def process_single_file(input_file, output_file, num_classes, method, roi_size,
                       min_component_size=100, max_components_per_class=3,
                       artifact_threshold=0.3, connectivity_threshold=0.7):
    """
    Process a single NIfTI file
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output NIfTI file
        num_classes: Number of classes
        method: Processing method
        roi_size: Size of ROI for processing
        min_component_size: Minimum size for valid components (MCA method)
        max_components_per_class: Maximum number of components per class (MCA method)
        artifact_threshold: Threshold for artifact detection (MCA method)
        connectivity_threshold: Threshold for connectivity analysis (MCA method)
    """
    try:
        # Load segmentation
        img = nib.load(input_file)
        segmentation = img.get_fdata()
        
        # Apply post-processing
        processed = advanced_ensemble_postprocessing(
            segmentation, num_classes, method, roi_size,
            min_component_size, max_components_per_class,
            artifact_threshold, connectivity_threshold
        )
        
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
                       choices=['ensemble', 'roi', 'center', 'fusion', 'combined', 'mca', 'mca_combined'],
                       default='ensemble',
                       help='Post-processing method (default: ensemble)')
    parser.add_argument('--roi_size', type=int, default=128,
                       help='Size of ROI for processing (default: 128)')
    parser.add_argument('--min_component_size', type=int, default=100,
                       help='Minimum size for valid components in MCA method (default: 100)')
    parser.add_argument('--max_components_per_class', type=int, default=3,
                       help='Maximum number of components per class in MCA method (default: 3)')
    parser.add_argument('--artifact_threshold', type=float, default=0.3,
                       help='Threshold for artifact detection in MCA method (default: 0.3)')
    parser.add_argument('--connectivity_threshold', type=float, default=0.7,
                       help='Threshold for connectivity analysis in MCA method (default: 0.7)')
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
    if args.method in ['mca', 'mca_combined']:
        print(f"Min component size: {args.min_component_size}")
        print(f"Max components per class: {args.max_components_per_class}")
        print(f"Artifact threshold: {args.artifact_threshold}")
        print(f"Connectivity threshold: {args.connectivity_threshold}")
    print("-" * 50)
    
    # Process files
    success_count = 0
    for input_file in input_files:
        output_file = args.output_folder / input_file.name
        if process_single_file(input_file, output_file, args.num_classes, args.method, args.roi_size,
                             args.min_component_size, args.max_components_per_class,
                             args.artifact_threshold, args.connectivity_threshold):
            success_count += 1
    
    print("-" * 50)
    print(f"Processing complete: {success_count}/{len(input_files)} files processed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())

# Basic MCA processing
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca

# # MCA with custom parameters for aggressive artifact removal
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca --min_component_size 200 --max_components_per_class 2 --artifact_threshold 0.2

# # MCA combined with other methods
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca_combined