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
                                             min_component_size=50, 
                                             max_components_per_class=1,
                                             artifact_threshold=0.6,
                                             connectivity_threshold=0.3):
    """
    Multiple Component Analysis (MCA) post-processing for artifact reduction
    
    This method analyzes components within each class and selects only the best
    (largest) component for each class to remove artifacts and noise.
    Special handling for class 1 (aorta/esophagus) to preserve connected elements.
    
    Args:
        segmentation: Input segmentation array
        num_classes: Number of classes
        min_component_size: Minimum size for valid components
        max_components_per_class: Maximum number of components to keep per class (default: 1 - only best)
        artifact_threshold: Threshold for artifact detection
        connectivity_threshold: Threshold for component connectivity analysis
    
    Returns:
        Processed segmentation with only the best component per class
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Special handling for Class 1 (esophagus) - skip MCA entirely
        if class_id == 1:
            # For esophagus: skip MCA and use minimal processing to preserve all components
            cleaned_mask = minimal_esophagus_processing(mask)
        else:
            # Standard MCA for other classes
            cleaned_mask = mca_artifact_reduction(
                mask, 
                min_component_size=min_component_size,
                max_components=1,  # Always select only the best component
                artifact_threshold=artifact_threshold,
                connectivity_threshold=connectivity_threshold
            )
        
        # Fallback: if MCA removes everything, keep the largest component
        if np.sum(cleaned_mask) == 0 and np.sum(mask) > 0:
            print(f"Warning: MCA removed all components for class {class_id}, keeping largest component")
            cleaned_mask = keep_largest_component(mask)
        
        processed[processed == class_id] = 0  # Clear original class
        processed[cleaned_mask] = class_id    # Set cleaned class
    
    return processed


def keep_largest_component(mask):
    """
    Keep only the largest connected component as fallback
    """
    if np.sum(mask) == 0:
        return mask
    
    labeled = label(mask)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Keep only the largest component
    largest_region = max(regions, key=lambda x: x.area)
    return (labeled == largest_region.label)


def minimal_esophagus_processing(mask):
    """
    Minimal processing for esophagus (Class 1) - skip MCA entirely and use only basic cleaning.
    This preserves the original segmentation quality while doing minimal improvements.
    """
    if np.sum(mask) == 0:
        return mask
    
    print(f"Minimal esophagus processing: {np.sum(mask)} voxels")
    
    # Step 1: Only remove single-pixel noise (extremely conservative)
    cleaned = remove_small_objects(mask, min_size=2)  # Only remove isolated pixels
    
    # Step 2: Fill only obvious holes (single iteration)
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Very gentle closing to connect only adjacent pixels
    if len(mask.shape) == 3:
        selem = ball(1)  # Smallest possible structuring element
    else:
        selem = disk(1)
    
    # Single gentle closing operation
    cleaned = binary_closing(cleaned, footprint=selem)
    
    # Step 4: Keep ALL components - no filtering based on size or shape
    # This preserves the original segmentation structure
    
    print(f"  Final voxels: {np.sum(cleaned)} (minimal changes)")
    
    return cleaned


def mca_artifact_reduction(mask, min_component_size=50, max_components=1,
                          artifact_threshold=0.6, connectivity_threshold=0.3):
    """
    Multiple Component Analysis for artifact reduction with connectivity preservation
    
    Args:
        mask: Binary mask for a single class
        min_component_size: Minimum size for valid components
        max_components: Maximum number of components to keep
        artifact_threshold: Threshold for artifact detection
        connectivity_threshold: Threshold for connectivity analysis
    
    Returns:
        Cleaned binary mask with artifacts removed but connected elements preserved
    """
    if np.sum(mask) == 0:
        return mask
    
    # Ensure mask is binary
    mask = mask.astype(bool)
    
    # Step 1: Fill holes first to connect nearby regions
    mask_filled = ndimage.binary_fill_holes(mask)
    
    # Step 2: Apply gentle morphological operations to connect nearby components
    if len(mask.shape) == 3:
        selem = ball(1)  # Conservative structuring element
    else:
        selem = disk(1)  # Conservative structuring element
    
    # Gentle closing to connect nearby regions without creating artifacts
    mask_connected = binary_closing(mask_filled, footprint=selem)
    
    # Step 3: Label connected components on the connected mask
    labeled = label(mask_connected)
    regions = regionprops(labeled)
    
    if len(regions) == 0:
        return mask
    
    # Step 4: Analyze component properties
    component_scores = []
    valid_components = []
    
    print(f"MCA: Analyzing {len(regions)} components for class (after connectivity enhancement)")
    
    for region in regions:
        # Calculate component features
        features = analyze_component_features(region, mask.shape)
        
        # Calculate artifact score (lower is better)
        artifact_score = calculate_artifact_score(features, artifact_threshold)
        
        print(f"  Component {region.label}: area={region.area}, artifact_score={artifact_score:.3f}, connectivity={features['connectivity']:.3f}")
        
        # Check if component meets minimum size requirement
        if region.area >= min_component_size:
            component_scores.append((region.label, artifact_score, features))
            valid_components.append(region.label)
        else:
            print(f"    -> Rejected: too small (area={region.area} < {min_component_size})")
    
    if not valid_components:
        print("MCA: No valid components found, returning empty mask")
        return np.zeros_like(mask, dtype=bool)
    
    print(f"MCA: {len(valid_components)} valid components found")
    
    # Step 5: Sort components by size first (largest first), then by artifact score
    component_scores.sort(key=lambda x: (-x[2]['area'], x[1]))  # -area for descending order
    
    # Step 6: Select the best component (largest with acceptable artifact score)
    selected_components = []
    for i, (comp_label, score, features) in enumerate(component_scores):
        if i >= max_components:
            print(f"    -> Rejected: too many components (limit={max_components})")
            break
        
        # Check if component meets minimum size and has acceptable artifact score
        if features['area'] >= min_component_size and score <= artifact_threshold:
            selected_components.append(comp_label)
            print(f"    -> Selected: component {comp_label} (area={features['area']}, score={score:.3f}, connectivity={features['connectivity']:.3f})")
            break  # Only select the first (best) component
        else:
            if features['area'] < min_component_size:
                print(f"    -> Rejected: too small (area={features['area']} < {min_component_size})")
            if score > artifact_threshold:
                print(f"    -> Rejected: high artifact score (score={score:.3f} > {artifact_threshold})")
    
    # Fallback: if no components selected, select the largest one anyway
    if not selected_components and component_scores:
        print("MCA: No components passed all checks, selecting largest component anyway")
        largest_comp = component_scores[0]  # Already sorted by size
        selected_components.append(largest_comp[0])
        print(f"    -> Fallback selected: component {largest_comp[0]} (area={largest_comp[2]['area']}, score={largest_comp[1]:.3f})")
    
    # Step 7: Create cleaned mask from connected components
    cleaned_mask = np.zeros_like(mask, dtype=bool)
    for component_label in selected_components:
        cleaned_mask[labeled == component_label] = True
    
    # Step 8: Apply final morphological cleaning (without hole filling)
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
    
    try:
        # Basic geometric features (always available)
        features['area'] = region.area
        features['solidity'] = region.solidity
        features['extent'] = region.extent
        
        # Handle 2D vs 3D specific properties
        if len(image_shape) == 2:  # 2D image
            features['perimeter'] = region.perimeter
            features['eccentricity'] = region.eccentricity
            features['compactness'] = (region.perimeter ** 2) / (4 * np.pi * region.area) if region.area > 0 else float('inf')
            features['aspect_ratio'] = region.major_axis_length / region.minor_axis_length if region.minor_axis_length > 0 else float('inf')
        else:  # 3D image - some properties not available
            features['perimeter'] = 0  # Not applicable for 3D
            features['eccentricity'] = 0.5  # Default value for 3D
            features['compactness'] = 1.0  # Default value for 3D
            features['aspect_ratio'] = 1.0  # Default value for 3D
        
        # Spatial features with error handling
        try:
            features['centroid_distance'] = calculate_centroid_distance(region, image_shape)
        except Exception as e:
            print(f"Warning: Error calculating centroid distance: {e}")
            features['centroid_distance'] = 0.5  # Default middle value
        
        try:
            features['boundary_ratio'] = calculate_boundary_ratio(region, image_shape)
        except Exception as e:
            print(f"Warning: Error calculating boundary ratio: {e}")
            features['boundary_ratio'] = 0.0  # Default no boundary touch
        
        # Connectivity features
        features['connectivity'] = calculate_connectivity_score(region, image_shape)
        
        # Density features
        features['density'] = region.area / region.bbox_area if region.bbox_area > 0 else 0
        
    except Exception as e:
        print(f"Warning: Error analyzing component features: {e}")
        # Return default features if analysis fails
        features = {
            'area': region.area if hasattr(region, 'area') else 0,
            'perimeter': 0,  # Default for both 2D and 3D
            'eccentricity': 0.5,  # Default for both 2D and 3D
            'solidity': region.solidity if hasattr(region, 'solidity') else 0.5,
            'extent': region.extent if hasattr(region, 'extent') else 0.5,
            'compactness': 1.0,  # Default for both 2D and 3D
            'aspect_ratio': 1.0,  # Default for both 2D and 3D
            'centroid_distance': 0.5,
            'boundary_ratio': 0.0,
            'connectivity': 0.5,
            'density': 0.5
        }
    
    return features


def calculate_artifact_score(features, threshold=0.6):
    """
    Calculate artifact score for a component (0-1, lower is better)
    
    Args:
        features: Dictionary of component features
        threshold: Threshold for artifact detection
    
    Returns:
        Artifact score (0-1)
    """
    score = 0.0
    
    # Shape-based artifact indicators (less aggressive)
    if features['eccentricity'] > 0.95:  # Very elongated (increased threshold)
        score += 0.15  # Reduced penalty
    if features['solidity'] < 0.3:  # Very irregular (reduced threshold)
        score += 0.15  # Reduced penalty
    if features['perimeter'] > 0 and features['compactness'] > 3.0:  # Very non-compact (increased threshold)
        score += 0.1   # Reduced penalty
    if features['aspect_ratio'] > 8.0:  # Very elongated (increased threshold)
        score += 0.1   # Reduced penalty
    
    # Size-based artifact indicators (less aggressive)
    if features['area'] < 20:  # Very small (reduced threshold)
        score += 0.2   # Reduced penalty
    if features['density'] < 0.2:  # Very sparse (reduced threshold)
        score += 0.05  # Reduced penalty
    
    # Spatial-based artifact indicators (less aggressive)
    if features['centroid_distance'] > 0.9:  # Far from center (increased threshold)
        score += 0.05  # Reduced penalty
    if features['boundary_ratio'] > 0.8:  # Too close to boundary (increased threshold)
        score += 0.05  # Reduced penalty
    
    # Connectivity-based artifact indicators (less aggressive)
    if features['connectivity'] < 0.2:  # Poor connectivity (reduced threshold)
        score += 0.1   # Reduced penalty
    
    return min(score, 1.0)


def calculate_centroid_distance(region, image_shape):
    """
    Calculate normalized distance from component centroid to image center
    """
    image_center = np.array(image_shape) / 2
    centroid = np.array(region.centroid)
    
    # Ensure centroid and image_center have the same dimensions
    if len(centroid) != len(image_center):
        # Pad with zeros if needed
        min_len = min(len(centroid), len(image_center))
        centroid = centroid[:min_len]
        image_center = image_center[:min_len]
    
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
    
    # Handle different dimensions properly
    if len(image_shape) == 2:  # 2D image
        # bbox format: (min_row, min_col, max_row, max_col)
        bbox_starts = [bbox[0], bbox[1]]  # min_row, min_col
        bbox_ends = [bbox[2], bbox[3]]    # max_row, max_col
        image_sizes = list(image_shape)
    elif len(image_shape) == 3:  # 3D image
        # bbox format: (min_row, min_col, min_slice, max_row, max_col, max_slice)
        bbox_starts = [bbox[0], bbox[1], bbox[2]]  # min_row, min_col, min_slice
        bbox_ends = [bbox[3], bbox[4], bbox[5]]    # max_row, max_col, max_slice
        image_sizes = list(image_shape)
    else:
        # Fallback for other dimensions
        bbox_starts = bbox[::2]
        bbox_ends = bbox[1::2]
        image_sizes = list(image_shape)
    
    # Check if component touches any image boundary
    for start, end, size in zip(bbox_starts, bbox_ends, image_sizes):
        if start <= 1 or end >= size - 1:
            boundary_touches += 1
    
    return boundary_touches / len(image_shape)


def calculate_connectivity_score(region, image_shape):
    """
    Calculate connectivity score based on component properties
    """
    # Higher solidity indicates better connectivity
    if len(image_shape) == 2:  # 2D image
        # Higher solidity and lower eccentricity indicate better connectivity
        connectivity = (region.solidity + (1 - region.eccentricity)) / 2
    else:  # 3D image - eccentricity not available
        # Use only solidity for 3D
        connectivity = region.solidity
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


def final_morphological_cleaning_with_holes(mask):
    """
    Apply final morphological operations for cleaning with conservative hole filling
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove very small objects
    cleaned = remove_small_objects(mask, min_size=20)
    
    # Fill holes once
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Apply gentle smoothing
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    
    # Gentle closing to connect nearby regions and fill small gaps
    cleaned = binary_closing(cleaned, footprint=selem)
    
    # Fill holes after closing
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Gentle opening to remove small protrusions
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def hole_filling_postprocessing(segmentation, num_classes=5):
    """
    Separate hole filling postprocessing method
    """
    processed = segmentation.copy().astype(np.int32)
    
    print(f"Hole filling postprocessing: input shape {segmentation.shape}")
    
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        original_voxels = np.sum(mask)
        
        if original_voxels == 0:
            continue
        
        # Fill holes for this class
        filled_mask = ndimage.binary_fill_holes(mask)
        filled_voxels = np.sum(filled_mask)
        
        print(f"  Class {class_id}: {original_voxels} -> {filled_voxels} voxels (change: {filled_voxels - original_voxels})")
        
        processed[processed == class_id] = 0  # Clear original class
        processed[filled_mask] = class_id     # Set filled class
    
    return processed


def aggressive_morphological_cleaning(segmentation, num_classes=5):
    """
    Apply aggressive morphological cleaning for MCA_combined method
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        
        if np.sum(mask) == 0:
            continue
        
        # Class-specific aggressive cleaning
        if class_id == 1:  # Esophagus - preserve small components
            cleaned = aggressive_esophagus_cleaning(mask)
        elif class_id == 2:  # Heart - ensure connectivity
            cleaned = aggressive_heart_cleaning(mask)
        elif class_id in [3, 4]:  # Lungs - fill holes and smooth
            cleaned = aggressive_lung_cleaning(mask)
        else:  # Other classes - standard aggressive cleaning
            cleaned = aggressive_standard_cleaning(mask)
        
        processed[processed == class_id] = 0  # Clear original class
        processed[cleaned] = class_id         # Set cleaned class
    
    return processed


def aggressive_esophagus_cleaning(mask):
    """
    Aggressive cleaning specifically for esophagus (Class 1)
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove very small objects (but keep smaller than other organs)
    cleaned = remove_small_objects(mask, min_size=5)
    
    # Fill holes multiple times
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Use smaller structuring element for esophagus
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    
    # Multiple rounds of closing to connect fragmented parts
    for _ in range(2):
        cleaned = binary_closing(cleaned, footprint=selem)
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Gentle opening to remove small protrusions
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def aggressive_heart_cleaning(mask):
    """
    Aggressive cleaning specifically for heart (Class 2)
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove small objects
    cleaned = remove_small_objects(mask, min_size=100)
    
    # Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Use larger structuring element for heart
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Aggressive closing to connect heart chambers
    cleaned = binary_closing(cleaned, footprint=selem)
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Keep only largest component (heart should be connected)
    labeled = label(cleaned)
    regions = regionprops(labeled)
    if len(regions) > 0:
        largest_region = max(regions, key=lambda x: x.area)
        cleaned = (labeled == largest_region.label)
    
    return cleaned


def aggressive_lung_cleaning(mask):
    """
    Aggressive cleaning specifically for lungs (Classes 3, 4)
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove small objects
    cleaned = remove_small_objects(mask, min_size=200)
    
    # Fill holes multiple times
    for _ in range(3):
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Use medium structuring element for lungs
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Aggressive closing to fill lung cavities
    cleaned = binary_closing(cleaned, footprint=selem)
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Gentle opening to smooth boundaries
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def aggressive_standard_cleaning(mask):
    """
    Aggressive cleaning for other classes
    """
    if np.sum(mask) == 0:
        return mask
    
    # Remove small objects
    cleaned = remove_small_objects(mask, min_size=50)
    
    # Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Use standard structuring element
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Aggressive closing
    cleaned = binary_closing(cleaned, footprint=selem)
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Aggressive opening
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned




def hd_optimized_postprocessing(segmentation, num_classes=5):
    """
    HD-optimized post-processing focused on reducing Hausdorff Distance metrics.
    Uses aggressive smoothing, boundary refinement, and morphological operations
    to create smoother, more regular boundaries.
    """
    processed = segmentation.copy().astype(np.int32)
    
    print("HD-optimized postprocessing: Focusing on boundary smoothing and HD reduction")
    
    for class_id in range(1, num_classes):
        mask = (segmentation == class_id)
        original_voxels = np.sum(mask)
        
        if original_voxels == 0:
            continue
        
        print(f"Processing Class {class_id}: {original_voxels} voxels")
        
        # Apply HD-optimized processing for each class
        if class_id == 1:  # Esophagus - gentle smoothing to preserve structure
            cleaned_mask = hd_optimized_esophagus_processing(mask)
        elif class_id == 2:  # Heart - aggressive smoothing for connectivity
            cleaned_mask = hd_optimized_heart_processing(mask)
        elif class_id == 3:  # Trachea - moderate smoothing
            cleaned_mask = hd_optimized_trachea_processing(mask)
        elif class_id == 4:  # Aorta - moderate smoothing
            cleaned_mask = hd_optimized_aorta_processing(mask)
        else:
            cleaned_mask = hd_optimized_standard_processing(mask)
        
        final_voxels = np.sum(cleaned_mask)
        print(f"  Class {class_id}: {original_voxels} -> {final_voxels} voxels (change: {final_voxels - original_voxels})")
        
        processed[processed == class_id] = 0  # Clear original class
        processed[cleaned_mask] = class_id    # Set cleaned class
    
    return processed


def hd_optimized_esophagus_processing(mask):
    """
    HD-optimized processing for esophagus (Class 1) - gentle smoothing
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Remove small noise
    cleaned = remove_small_objects(mask, min_size=5)
    
    # Step 2: Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Gentle smoothing with small structuring element
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    
    # Multiple rounds of gentle closing for smooth boundaries
    for _ in range(2):
        cleaned = binary_closing(cleaned, footprint=selem)
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Gentle opening to remove small protrusions
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def hd_optimized_heart_processing(mask):
    """
    HD-optimized processing for heart (Class 2) - aggressive smoothing for connectivity
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Remove small objects
    cleaned = remove_small_objects(mask, min_size=50)
    
    # Step 2: Fill holes multiple times
    for _ in range(3):
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Aggressive smoothing with larger structuring element
    if len(mask.shape) == 3:
        selem = ball(3)  # Larger structuring element for more smoothing
    else:
        selem = disk(3)
    
    # Multiple rounds of aggressive closing
    for _ in range(3):
        cleaned = binary_closing(cleaned, footprint=selem)
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Keep only largest component (heart should be connected)
    labeled = label(cleaned)
    regions = regionprops(labeled)
    if len(regions) > 0:
        largest_region = max(regions, key=lambda x: x.area)
        cleaned = (labeled == largest_region.label)
    
    # Final gentle opening
    if len(mask.shape) == 3:
        selem = ball(1)
    else:
        selem = disk(1)
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def hd_optimized_trachea_processing(mask):
    """
    HD-optimized processing for trachea (Class 3) - moderate smoothing
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Remove small objects
    cleaned = remove_small_objects(mask, min_size=20)
    
    # Step 2: Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Moderate smoothing
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Multiple rounds of moderate closing
    for _ in range(2):
        cleaned = binary_closing(cleaned, footprint=selem)
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Moderate opening
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def hd_optimized_aorta_processing(mask):
    """
    HD-optimized processing for aorta (Class 4) - moderate smoothing
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Remove small objects
    cleaned = remove_small_objects(mask, min_size=20)
    
    # Step 2: Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Moderate smoothing
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Multiple rounds of moderate closing
    for _ in range(2):
        cleaned = binary_closing(cleaned, footprint=selem)
        cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Moderate opening
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def hd_optimized_standard_processing(mask):
    """
    HD-optimized processing for other classes - standard smoothing
    """
    if np.sum(mask) == 0:
        return mask
    
    # Step 1: Remove small objects
    cleaned = remove_small_objects(mask, min_size=30)
    
    # Step 2: Fill holes
    cleaned = ndimage.binary_fill_holes(cleaned)
    
    # Step 3: Standard smoothing
    if len(mask.shape) == 3:
        selem = ball(2)
    else:
        selem = disk(2)
    
    # Standard closing and opening
    cleaned = binary_closing(cleaned, footprint=selem)
    cleaned = ndimage.binary_fill_holes(cleaned)
    cleaned = binary_opening(cleaned, footprint=selem)
    
    return cleaned


def test_class1_only_postprocessing(segmentation, num_classes=5, method='mca', roi_size=128,
                                   min_component_size=100, max_components_per_class=3,
                                   artifact_threshold=0.3, connectivity_threshold=0.7):
    """
    Test method that only processes Class 1 (esophagus) while keeping other classes unchanged.
    This saves compute time for testing Class 1 improvements.
    """
    processed = segmentation.copy().astype(np.int32)
    
    # Only process Class 1 (esophagus)
    class_id = 1
    mask = (segmentation == class_id)
    
    if np.sum(mask) > 0:
        print(f"Testing Class {class_id} (esophagus) processing...")
        
        if method == 'mca':
            # Apply minimal processing for Class 1 (skip MCA entirely)
            cleaned_mask = minimal_esophagus_processing(mask)
            
            # Fallback: if MCA removes everything, keep the largest component
            if np.sum(cleaned_mask) == 0 and np.sum(mask) > 0:
                print(f"Warning: MCA removed all components for class {class_id}, keeping largest component")
                cleaned_mask = keep_largest_component(mask)
            
            processed[processed == class_id] = 0  # Clear original class
            processed[cleaned_mask] = class_id    # Set cleaned class
            
        elif method == 'mca_combined':
            # Apply aggressive processing only to Class 1
            # Work directly on the Class 1 mask instead of the full segmentation
            
            # Step 1: Basic morphological operations
            cleaned_mask = mask.copy()
            
            # Step 2: Fill holes
            cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
            
            # Step 3: Gentle morphological operations
            if len(mask.shape) == 3:
                selem = ball(1)
            else:
                selem = disk(1)
            
            # Gentle closing
            cleaned_mask = binary_closing(cleaned_mask, footprint=selem)
            
            # Step 4: Remove small objects (conservative)
            cleaned_mask = remove_small_objects(cleaned_mask, min_size=10)
            
            # Step 5: Final hole filling
            cleaned_mask = ndimage.binary_fill_holes(cleaned_mask)
            
            processed[processed == class_id] = 0  # Clear original class
            processed[cleaned_mask] = class_id     # Set cleaned class
        
        print(f"Class {class_id} processing complete. Voxels: {np.sum(mask)} -> {np.sum(processed == class_id)}")
    else:
        print(f"Class {class_id} (esophagus) not found in segmentation")
    
    return processed


def enhanced_ensemble_voting_postprocessing(segmentation, num_classes=5):
    """
    Enhanced ensemble voting with more aggressive parameters
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (processed == class_id)
        if np.sum(mask) == 0:
            continue
            
        # More aggressive morphological operations
        if len(mask.shape) == 3:
            selem = ball(2)  # Larger structuring element
        else:
            selem = disk(2)
        
        # Aggressive closing to connect components
        cleaned = binary_closing(mask, footprint=selem)
        # Aggressive opening to remove noise
        cleaned = binary_opening(cleaned, footprint=selem)
        
        processed[processed == class_id] = 0
        processed[cleaned] = class_id
    
    return processed


def enhanced_roi_based_postprocessing(segmentation, num_classes=5, roi_size=64):
    """
    Enhanced ROI-based processing with smaller, more focused ROIs
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (processed == class_id)
        if np.sum(mask) == 0:
            continue
        
        # Find connected components
        labeled = label(mask)
        regions = regionprops(labeled)
        
        for region in regions:
            # Get bounding box
            minr, minc, mins, maxr, maxc, maxs = region.bbox
            
            # Create smaller, more focused ROI
            center_r, center_c, center_s = region.centroid
            half_size = roi_size // 2
            
            # Extract ROI with more aggressive processing
            roi_start_r = max(0, int(center_r) - half_size)
            roi_end_r = min(mask.shape[0], int(center_r) + half_size)
            roi_start_c = max(0, int(center_c) - half_size)
            roi_end_c = min(mask.shape[1], int(center_c) + half_size)
            roi_start_s = max(0, int(center_s) - half_size)
            roi_end_s = min(mask.shape[2], int(center_s) + half_size)
            
            roi_mask = mask[roi_start_r:roi_end_r, roi_start_c:roi_end_c, roi_start_s:roi_end_s]
            
            if np.sum(roi_mask) > 0:
                # Aggressive processing within ROI
                if len(roi_mask.shape) == 3:
                    selem = ball(1)
                else:
                    selem = disk(1)
                
                # Multiple iterations of closing and opening
                for _ in range(2):
                    roi_mask = binary_closing(roi_mask, footprint=selem)
                    roi_mask = binary_opening(roi_mask, footprint=selem)
                
                # Update the processed segmentation
                processed[roi_start_r:roi_end_r, roi_start_c:roi_end_c, roi_start_s:roi_end_s][roi_mask] = class_id
    
    return processed


def enhanced_center_constraint_postprocessing(segmentation, num_classes=5):
    """
    Enhanced center constraint with stricter constraints
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (processed == class_id)
        if np.sum(mask) == 0:
            continue
        
        # Find center of mass
        center = np.array(ndimage.center_of_mass(mask))
        
        # Calculate distance from center for each voxel
        coords = np.where(mask)
        distances = np.sqrt(np.sum((np.array(coords).T - center)**2, axis=1))
        
        # More aggressive distance threshold (keep only closer voxels)
        max_distance = np.percentile(distances, 75)  # Keep only 75% closest to center
        
        # Create distance mask
        distance_mask = np.zeros_like(mask, dtype=bool)
        for i, (r, c, s) in enumerate(zip(coords[0], coords[1], coords[2])):
            if distances[i] <= max_distance:
                distance_mask[r, c, s] = True
        
        processed[processed == class_id] = 0
        processed[distance_mask] = class_id
    
    return processed


def enhanced_multi_label_fusion_postprocessing(segmentation, num_classes=5):
    """
    Enhanced multi-label fusion with more aggressive fusion
    """
    processed = segmentation.copy().astype(np.int32)
    
    # Create probability maps for each class
    prob_maps = {}
    for class_id in range(1, num_classes):
        mask = (processed == class_id)
        if np.sum(mask) > 0:
            # Apply Gaussian smoothing for more aggressive fusion
            prob_maps[class_id] = ndimage.gaussian_filter(mask.astype(float), sigma=2.0)
        else:
            prob_maps[class_id] = np.zeros_like(processed, dtype=float)
    
    # Fusion with more aggressive thresholding
    fused = np.zeros_like(processed)
    for class_id in range(1, num_classes):
        if class_id in prob_maps:
            # Lower threshold for more aggressive fusion
            threshold = np.percentile(prob_maps[class_id][prob_maps[class_id] > 0], 25)
            class_mask = prob_maps[class_id] > threshold
            fused[class_mask] = class_id
    
    return fused


def final_aggressive_cleanup(segmentation, num_classes=5):
    """
    Final aggressive morphological cleanup
    """
    processed = segmentation.copy().astype(np.int32)
    
    for class_id in range(1, num_classes):
        mask = (processed == class_id)
        if np.sum(mask) == 0:
            continue
        
        # Aggressive morphological operations
        if len(mask.shape) == 3:
            selem = ball(1)
        else:
            selem = disk(1)
        
        # Multiple iterations of aggressive operations
        cleaned = mask.copy()
        for _ in range(3):
            cleaned = binary_closing(cleaned, footprint=selem)
            cleaned = binary_opening(cleaned, footprint=selem)
        
        # Remove small objects aggressively
        cleaned = remove_small_objects(cleaned, min_size=20)
        
        # Fill holes
        cleaned = ndimage.binary_fill_holes(cleaned)
        
        processed[processed == class_id] = 0
        processed[cleaned] = class_id
    
    return processed


def mca_without_class3(segmentation, num_classes=5, min_component_size=50, 
                      max_components_per_class=3, artifact_threshold=0.6, 
                      connectivity_threshold=0.3):
    """
    MCA processing for classes 0,1,2,4 (preserving class 3/trachea)
    """
    processed = segmentation.copy()
    
    for class_id in range(num_classes):
        if class_id == 0:  # Skip background class - it should remain as background
            continue
        if class_id == 3:  # Preserve class 3 (trachea) without processing
            continue
            
        mask = (segmentation == class_id)
        if np.any(mask):
            if class_id == 1:  # Esophagus - use minimal processing
                cleaned_mask = minimal_esophagus_processing(mask)
            else:  # Other classes - use standard MCA
                cleaned_mask = mca_artifact_reduction(mask, min_component_size, max_components_per_class, 
                                                    artifact_threshold, connectivity_threshold)
            
            processed[processed == class_id] = 0  # Clear original class
            processed[cleaned_mask] = class_id    # Set cleaned class
    
    return processed


def mca_class3_only(segmentation, min_component_size=50, max_components_per_class=3, 
                   artifact_threshold=0.6, connectivity_threshold=0.3):
    """
    MCA processing for class 3 (trachea) only
    """
    processed = segmentation.copy()
    class_3_mask = (processed == 3)
    
    if np.any(class_3_mask):
        cleaned_mask = mca_artifact_reduction(class_3_mask, min_component_size, max_components_per_class, 
                                            artifact_threshold, connectivity_threshold)
        
        processed[processed == 3] = 0  # Clear original class 3
        processed[cleaned_mask] = 3     # Set cleaned class 3
    
    return processed


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
        processed = multiple_component_analysis_postprocessing(
            segmentation, num_classes, min_component_size, 
            max_components_per_class, artifact_threshold, connectivity_threshold
        )

        processed = hd_optimized_postprocessing(processed, num_classes)

        return processed

    elif method == 'combined':
        # Enhanced combined approach: Apply all methods sequentially with boosted effects
        print("Combined method: Applying enhanced ensemble + ROI + center + fusion processing")
        
        # Step 1: Enhanced ensemble voting with more aggressive parameters
        processed = enhanced_ensemble_voting_postprocessing(segmentation, num_classes)
        print(f"After enhanced ensemble: {np.sum(processed > 0)} total voxels")
        
        # Step 2: Enhanced ROI-based processing with smaller, more focused ROIs
        processed = enhanced_roi_based_postprocessing(processed, num_classes, roi_size // 2)
        print(f"After enhanced ROI: {np.sum(processed > 0)} total voxels")
        
        # Step 3: Enhanced center constraint with stricter constraints
        processed = enhanced_center_constraint_postprocessing(processed, num_classes)
        print(f"After enhanced center: {np.sum(processed > 0)} total voxels")
        
        # Step 4: Enhanced multi-label fusion with more aggressive fusion
        processed = enhanced_multi_label_fusion_postprocessing(processed, num_classes)
        print(f"After enhanced fusion: {np.sum(processed > 0)} total voxels")
        
        # Step 5: Final aggressive morphological cleanup
        processed = final_aggressive_cleanup(processed, num_classes)
        print(f"After final cleanup: {np.sum(processed > 0)} total voxels")
        
        return processed
    elif method == 'final':
        # Final optimized method: MCA on classes 0,1,2,4 -> HD optimized -> MCA on class 3 only
        print("Final method: MCA (classes 0,1,2,4) -> HD optimized -> MCA (class 3 only)")
        
        # Step 1: MCA on classes 0,1,2,4 (skip class 3)
        print("Step 1: Applying MCA to classes 0,1,2,4 (skipping class 3)")
        processed = mca_without_class3(segmentation, num_classes, min_component_size, 
                                     max_components_per_class, artifact_threshold, connectivity_threshold)
        print(f"After MCA (classes 0,1,2,4): {np.sum(processed > 0)} total voxels")
        
        # Step 2: HD optimized processing on all classes
        print("Step 2: Applying HD optimized processing to all classes")
        processed = hd_optimized_postprocessing(processed, num_classes)
        print(f"After HD optimized: {np.sum(processed > 0)} total voxels")
        
        # Step 3: MCA on class 3 only (trachea)
        print("Step 3: Applying MCA to class 3 (trachea) only")
        processed = mca_class3_only(processed, min_component_size, max_components_per_class, 
                                  artifact_threshold, connectivity_threshold)
        print(f"After final MCA (class 3): {np.sum(processed > 0)} total voxels")
        
        return processed

    elif method == 'hole_filling':
        # Separate hole filling method
        return hole_filling_postprocessing(segmentation, num_classes)
    elif method == 'hd_optimized':
        # HD-optimized approach focused on reducing Hausdorff Distance
        # Uses aggressive smoothing and boundary refinement
        return hd_optimized_postprocessing(segmentation, num_classes)
    else:
        raise ValueError(f"Unknown method: {method}")


def process_single_file(input_file, output_file, num_classes, method, roi_size,
                       min_component_size=100, max_components_per_class=3,
                       artifact_threshold=0.3, connectivity_threshold=0.7,
                       separate_classes=False):
    """
    Process a single NIfTI file
    
    Args:
        input_file: Path to input NIfTI file
        output_file: Path to output NIfTI file (or folder if separate_classes=True)
        num_classes: Number of classes
        method: Processing method
        roi_size: Size of ROI for processing
        min_component_size: Minimum size for valid components (MCA method)
        max_components_per_class: Maximum number of components per class (MCA method)
        artifact_threshold: Threshold for artifact detection (MCA method)
        connectivity_threshold: Threshold for connectivity analysis (MCA method)
        separate_classes: If True, save one file per class; if False, save combined file
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
        
        if separate_classes:
            # Save separate files for each class
            output_file.parent.mkdir(parents=True, exist_ok=True)
            base_name = input_file.stem.replace('.nii', '')  # Remove .nii.gz extension properly
            
            saved_files = 0
            for class_id in range(1, num_classes):
                # Create binary mask for this class
                class_mask = (processed == class_id).astype(np.int32)
                
                # Only save if the class has any voxels
                if np.sum(class_mask) > 0:
                    class_output_file = output_file.parent / f"{base_name}_class_{class_id}.nii.gz"
                    nib.save(nib.Nifti1Image(class_mask, img.affine, img.header), class_output_file)
                    saved_files += 1
                    print(f"  ✓ Saved class {class_id}: {class_output_file.name}")
            
            print(f"✓ Processed: {input_file.name} -> {saved_files} class files")
        else:
            # Save combined result (original behavior)
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
                       choices=['ensemble', 'roi', 'center', 'fusion', 'combined', 'mca', 'final',
                               'test_class1_mca', 'test_class1_combined', 'hole_filling', 'hd_optimized'],
                       default='ensemble',
                       help='Post-processing method (default: ensemble). test_class1_* methods only process Class 1 to save compute. hole_filling is a separate hole filling method. hd_optimized focuses on reducing HD metrics.')
    parser.add_argument('--roi_size', type=int, default=128,
                       help='Size of ROI for processing (default: 128)')
    parser.add_argument('--min_component_size', type=int, default=50,
                       help='Minimum size for valid components in MCA method (default: 50)')
    parser.add_argument('--max_components_per_class', type=int, default=1,
                       help='Maximum number of components per class in MCA method (default: 1 - only best component)')
    parser.add_argument('--artifact_threshold', type=float, default=0.6,
                       help='Threshold for artifact detection in MCA method (default: 0.6)')
    parser.add_argument('--connectivity_threshold', type=float, default=0.3,
                       help='Threshold for connectivity analysis in MCA method (default: 0.3)')
    parser.add_argument('--separate_classes', action='store_true', default=False,
                       help='Save separate files for each class (default: False - saves combined file)')
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
    # Determine output mode
    separate_classes = args.separate_classes
    print(f"Output mode: {'Separate files per class' if separate_classes else 'Combined file'}")
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
                             args.artifact_threshold, args.connectivity_threshold,
                             separate_classes):
            success_count += 1
    
    print("-" * 50)
    print(f"Processing complete: {success_count}/{len(input_files)} files processed successfully")
    
    return 0


if __name__ == "__main__":
    exit(main())

# Basic MCA processing
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca

# MCA with custom parameters (selects only the best/largest component per class)
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca --min_component_size 30 --artifact_threshold 0.8

# MCA combined with other methods
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca_combined

# MCA with very permissive settings (if still losing components)
# python ensemble_postprocessing.py --input_folder /path/to/input --output_folder /path/to/output --method mca --min_component_size 10 --artifact_threshold 0.9