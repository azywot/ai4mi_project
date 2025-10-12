#!/usr/bin/env python3
"""
Test script for MCA (Multiple Component Analysis) post-processing
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from ensemble_postprocessing import multiple_component_analysis_postprocessing

def create_test_segmentation():
    """
    Create a test segmentation with multiple components per class
    """
    # Create a 3D test volume
    shape = (64, 64, 32)
    segmentation = np.zeros(shape, dtype=np.int32)
    
    # Class 1 (Aorta): Multiple small components + one large
    segmentation[10:15, 10:15, 10:15] = 1  # Large component
    segmentation[20:22, 20:22, 20:22] = 1  # Small component
    segmentation[30:32, 30:32, 30:32] = 1  # Small component
    
    # Class 2 (Heart): One large component + noise
    segmentation[40:55, 40:55, 15:25] = 2  # Large component
    segmentation[45:47, 45:47, 26:28] = 2  # Small noise component
    
    # Class 3 (Left Lung): Multiple components
    segmentation[5:20, 5:20, 5:10] = 3     # Component 1
    segmentation[5:20, 5:20, 15:20] = 3    # Component 2
    segmentation[5:20, 5:20, 25:30] = 3    # Component 3
    
    # Class 4 (Right Lung): One large component
    segmentation[35:50, 35:50, 5:30] = 4   # Large component
    
    return segmentation

def test_mca_method():
    """
    Test the MCA method with different parameters
    """
    print("=== Testing MCA Method ===")
    
    # Create test segmentation
    test_seg = create_test_segmentation()
    print(f"Original segmentation shape: {test_seg.shape}")
    print(f"Original class counts:")
    for i in range(5):
        count = np.sum(test_seg == i)
        print(f"  Class {i}: {count} voxels")
    
    # Test with very permissive settings
    print("\n--- Test 1: Very permissive settings ---")
    result1 = multiple_component_analysis_postprocessing(
        test_seg, 
        num_classes=5,
        min_component_size=10,
        max_components_per_class=10,
        artifact_threshold=0.9,
        connectivity_threshold=0.1
    )
    
    print(f"Result 1 class counts:")
    for i in range(5):
        count = np.sum(result1 == i)
        print(f"  Class {i}: {count} voxels")
    
    # Test with moderate settings
    print("\n--- Test 2: Moderate settings ---")
    result2 = multiple_component_analysis_postprocessing(
        test_seg, 
        num_classes=5,
        min_component_size=50,
        max_components_per_class=5,
        artifact_threshold=0.6,
        connectivity_threshold=0.3
    )
    
    print(f"Result 2 class counts:")
    for i in range(5):
        count = np.sum(result2 == i)
        print(f"  Class {i}: {count} voxels")
    
    # Test with strict settings
    print("\n--- Test 3: Strict settings ---")
    result3 = multiple_component_analysis_postprocessing(
        test_seg, 
        num_classes=5,
        min_component_size=100,
        max_components_per_class=2,
        artifact_threshold=0.3,
        connectivity_threshold=0.7
    )
    
    print(f"Result 3 class counts:")
    for i in range(5):
        count = np.sum(result3 == i)
        print(f"  Class {i}: {count} voxels")
    
    # Save test results
    print("\n--- Saving test results ---")
    output_dir = Path("test_mca_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create dummy affine and header
    affine = np.eye(4)
    header = nib.Nifti1Header()
    
    # Save original
    nib.save(nib.Nifti1Image(test_seg, affine, header), output_dir / "original.nii.gz")
    print("✓ Saved original.nii.gz")
    
    # Save results
    nib.save(nib.Nifti1Image(result1, affine, header), output_dir / "result_permissive.nii.gz")
    print("✓ Saved result_permissive.nii.gz")
    
    nib.save(nib.Nifti1Image(result2, affine, header), output_dir / "result_moderate.nii.gz")
    print("✓ Saved result_moderate.nii.gz")
    
    nib.save(nib.Nifti1Image(result3, affine, header), output_dir / "result_strict.nii.gz")
    print("✓ Saved result_strict.nii.gz")
    
    print("\n=== Test completed ===")
    print("Check the test_mca_results/ folder for output files")

if __name__ == "__main__":
    test_mca_method()


