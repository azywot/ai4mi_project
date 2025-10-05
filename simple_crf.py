"""
Simplified Conditional Random Fields (CRFs) for Medical Image Segmentation Post-processing

This is a simplified, efficient implementation of CRFs for post-processing
segmentation predictions using probabilistic graphical modeling.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Optional
import argparse
from scipy import ndimage
import warnings


class SimpleCRF:
    """
    Simplified CRF implementation for segmentation post-processing.
    
    This CRF uses:
    1. Unary potentials from original predictions
    2. Pairwise potentials for spatial smoothness
    3. Mean field approximation for inference
    """
    
    def __init__(self, num_classes: int = 5, 
                 spatial_weight: float = 1.0,
                 num_iterations: int = 5):
        """
        Initialize the CRF.
        
        Args:
            num_classes: Number of segmentation classes
            spatial_weight: Weight for spatial smoothness
            num_iterations: Number of mean field iterations
        """
        self.num_classes = num_classes
        self.spatial_weight = spatial_weight
        self.num_iterations = num_iterations
    
    def process_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Apply CRF post-processing to segmentation.
        
        Args:
            segmentation: Input segmentation [H, W] or [H, W, D]
            
        Returns:
            Refined segmentation
        """
        if segmentation.ndim == 2:
            return self._process_2d(segmentation)
        elif segmentation.ndim == 3:
            return self._process_3d(segmentation)
        else:
            raise ValueError(f"Unsupported segmentation dimensions: {segmentation.ndim}")
    
    def _process_2d(self, segmentation: np.ndarray) -> np.ndarray:
        """Process 2D segmentation."""
        H, W = segmentation.shape
        
        # Convert to one-hot probabilities
        probs = self._segmentation_to_probs(segmentation)
        
        # Apply mean field approximation
        refined_probs = self._mean_field_2d(probs)
        
        # Convert back to segmentation
        refined_seg = np.argmax(refined_probs, axis=0)
        
        return refined_seg
    
    def _process_3d(self, segmentation: np.ndarray) -> np.ndarray:
        """Process 3D segmentation slice by slice."""
        H, W, D = segmentation.shape
        refined_seg = np.zeros_like(segmentation)
        
        print(f"Processing 3D volume with {D} slices...")
        
        for d in range(D):
            if d % 10 == 0:
                print(f"  Processing slice {d+1}/{D}")
            
            slice_seg = segmentation[:, :, d]
            refined_slice = self._process_2d(slice_seg)
            refined_seg[:, :, d] = refined_slice
        
        return refined_seg
    
    def _segmentation_to_probs(self, segmentation: np.ndarray) -> np.ndarray:
        """Convert segmentation to probability distribution."""
        H, W = segmentation.shape
        probs = np.zeros((self.num_classes, H, W))
        
        for c in range(self.num_classes):
            probs[c, :, :] = (segmentation == c).astype(np.float32)
        
        # Add small epsilon to avoid numerical issues
        probs = np.maximum(probs, 1e-8)
        
        # Normalize
        probs = probs / np.sum(probs, axis=0, keepdims=True)
        
        return probs
    
    def _mean_field_2d(self, probs: np.ndarray) -> np.ndarray:
        """Apply mean field approximation for 2D CRF."""
        C, H, W = probs.shape
        
        # Initialize
        Q = probs.copy()
        
        for iteration in range(self.num_iterations):
            Q_old = Q.copy()
            
            # Compute pairwise potentials
            pairwise = self._compute_pairwise_2d(Q)
            
            # Update Q
            Q = probs + self.spatial_weight * pairwise
            Q = np.exp(Q)
            Q = Q / np.sum(Q, axis=0, keepdims=True)
            
            # Check convergence
            if np.allclose(Q, Q_old, atol=1e-6):
                break
        
        return Q
    
    def _compute_pairwise_2d(self, Q: np.ndarray) -> np.ndarray:
        """Compute pairwise potentials for 2D CRF."""
        C, H, W = Q.shape
        pairwise = np.zeros_like(Q)
        
        # 4-connected neighborhood
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dy, dx in directions:
            Q_shifted = np.zeros_like(Q)
            
            if dy == 1:  # Down
                Q_shifted[:, :-1, :] = Q[:, 1:, :]
            elif dy == -1:  # Up
                Q_shifted[:, 1:, :] = Q[:, :-1, :]
            elif dx == 1:  # Right
                Q_shifted[:, :, :-1] = Q[:, :, 1:]
            elif dx == -1:  # Left
                Q_shifted[:, :, 1:] = Q[:, :, :-1]
            
            # Add spatial smoothness
            for c in range(C):
                pairwise[c, :, :] += Q_shifted[c, :, :]
        
        return pairwise


def process_folder_with_simple_crf(input_folder: Path, output_folder: Path,
                                 num_classes: int = 5,
                                 spatial_weight: float = 1.0,
                                 num_iterations: int = 5) -> None:
    """
    Process all NIfTI files in a folder with simple CRF post-processing.
    
    Args:
        input_folder: Folder containing input NIfTI files
        output_folder: Folder to save processed files
        num_classes: Number of segmentation classes
        spatial_weight: Weight for spatial smoothness
        num_iterations: Number of mean field iterations
    """
    # Create output folder
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Initialize CRF
    crf = SimpleCRF(
        num_classes=num_classes,
        spatial_weight=spatial_weight,
        num_iterations=num_iterations
    )
    
    # Process all NIfTI files
    nifti_files = list(input_folder.glob("*.nii.gz"))
    
    if not nifti_files:
        print(f"No NIfTI files found in {input_folder}")
        return
    
    print(f"Processing {len(nifti_files)} files with simple CRF post-processing...")
    
    for input_file in nifti_files:
        output_file = output_folder / input_file.name
        
        # Load segmentation
        seg_img = nib.load(str(input_file))
        seg_data = seg_img.get_fdata()
        affine = seg_img.affine
        header = seg_img.header
        
        # Apply CRF post-processing
        refined_data = crf.process_segmentation(seg_data)
        
        # Save processed NIfTI file
        output_nii = nib.Nifti1Image(refined_data.astype(np.uint8), affine, header)
        nib.save(output_nii, str(output_file))
        
        print(f"Processed {input_file.name} -> {output_file.name}")
    
    print(f"Completed processing. Results saved to {output_folder}")


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Simple CRF Post-processing for Medical Image Segmentation')
    
    parser.add_argument('--input_folder', type=Path, required=True,
                        help='Input folder containing NIfTI files')
    parser.add_argument('--output_folder', type=Path, required=True,
                        help='Output folder for processed files')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='Number of segmentation classes (default: 5)')
    parser.add_argument('--spatial_weight', type=float, default=1.0,
                        help='Weight for spatial smoothness (default: 1.0)')
    parser.add_argument('--num_iterations', type=int, default=5,
                        help='Number of mean field iterations (default: 5)')
    
    args = parser.parse_args()
    
    # Process files
    process_folder_with_simple_crf(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        num_classes=args.num_classes,
        spatial_weight=args.spatial_weight,
        num_iterations=args.num_iterations
    )


if __name__ == "__main__":
    main()
