#!/usr/bin/env python3
"""
Test script for improved MCA method with connectivity preservation
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from ensemble_postprocessing import multiple_component_analysis_postprocessing

def create_test_segmentation_with_gaps():
    """
    Create a test segmentation with gaps that should be connected (like aorta/esophagus)
    """
    # Create a 3D test volume
    shape = (64, 64, 32)
    segmentation = np.zeros(shape, dtype=np.int32)
    
    # Class 1 (Aorta/Esophagus): Create a structure with gaps that should be connected
    # Main body
    segmentation[10:15, 10:15, 10:15] = 1  # Main body
    segmentation[10:15, 10:15, 16:18] = 1  # Small gap
    segmentation[10:15, 10:15, 19:22] = 1  # Another part
    segmentation[10:15, 10:15, 23:25] = 1  # Another gap
    segmentation[10:15, 10:15, 26:28] = 1  # Final part
    
    # Add some disconnected noise
    segmentation[20:22, 20:22, 20:22] = 1  # Small disconnected component
    segmentation[30:32, 30:32, 30:32] = 1  # Another small disconnected component
    
    # Class 2 (Heart): One large component
    segmentation[40:55, 40:55, 15:25] = 2  # Large component
    
    # Class 3 (Left Lung): Multiple components
    segmentation[5:20, 5:20, 5:10] = 3     # Component 1
    segmentation[5:20, 5:20, 15:20] = 3    # Component 2
    segmentation[5:20, 5:20, 25:30] = 3    # Component 3
    
    # Class 4 (Right Lung): One large component
    segmentation[35:50, 35:50, 5:30] = 4   # Large component
    
    return segmentation

def test_improved_mca():
    """
    Test the improved MCA method with connectivity preservation
    """
    print("=== Testing Improved MCA Method with Connectivity Preservation ===")
    
    # Create test segmentation with gaps
    test_seg = create_test_segmentation_with_gaps()
    print(f"Original segmentation shape: {test_seg.shape}")
    print(f"Original class counts:")
    for i in range(5):
        count = np.sum(test_seg == i)
        print(f"  Class {i}: {count} voxels")
    
    # Test with moderate settings
    print("\n--- Test: Improved MCA with connectivity preservation ---")
    result = multiple_component_analysis_postprocessing(
        test_seg, 
        num_classes=5,
        min_component_size=50,
        max_components_per_class=1,
        artifact_threshold=0.6,
        connectivity_threshold=0.3
    )
    
    print(f"Result class counts:")
    for i in range(5):
        count = np.sum(result == i)
        print(f"  Class {i}: {count} voxels")
    
    # Calculate improvement for class 1 (aorta/esophagus)
    original_class1 = np.sum(test_seg == 1)
    result_class1 = np.sum(result == 1)
    improvement = result_class1 / original_class1 if original_class1 > 0 else 0
    
    print(f"\nClass 1 (Aorta/Esophagus) improvement:")
    print(f"  Original: {original_class1} voxels")
    print(f"  Result: {result_class1} voxels")
    print(f"  Preservation ratio: {improvement:.2%}")
    
    if improvement > 0.8:
        print("  ✅ Good: Most of the connected structure is preserved")
    elif improvement > 0.5:
        print("  ⚠️  Moderate: Some connected elements may be lost")
    else:
        print("  ❌ Poor: Too many connected elements are being removed")
    
    # Save test results
    print("\n--- Saving test results ---")
    output_dir = Path("test_mca_connectivity_results")
    output_dir.mkdir(exist_ok=True)
    
    # Create dummy affine and header
    affine = np.eye(4)
    header = nib.Nifti1Header()
    
    # Save original
    nib.save(nib.Nifti1Image(test_seg, affine, header), output_dir / "original_with_gaps.nii.gz")
    print("✓ Saved original_with_gaps.nii.gz")
    
    # Save result
    nib.save(nib.Nifti1Image(result, affine, header), output_dir / "result_improved_mca.nii.gz")
    print("✓ Saved result_improved_mca.nii.gz")
    
    print("\n=== Test completed ===")
    print("Check the test_mca_connectivity_results/ folder for output files")
    print("The improved MCA should preserve more of the connected structure for class 1")

if __name__ == "__main__":
    test_improved_mca()

