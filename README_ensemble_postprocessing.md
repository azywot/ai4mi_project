# Ensemble Post-Processing for Medical Images

These methods are specifically designed for medical image segmentation and provide significant improvements over basic post-processing.

## 🚀 Quick Start

### 1. Run Individual Methods
```bash
# Ensemble voting (strongest effect)
python ensemble_postprocessing.py \
    --input_folder val2/pred \
    --output_folder val2/pred_ensemble \
    --num_classes 5 \
    --method ensemble

# Combined processing (maximum quality)
python ensemble_postprocessing.py \
    --input_folder val2/pred \
    --output_folder val2/pred_combined_ensemble \
    --num_classes 5 \
    --method combined \
    --roi_size 128
```

## 📊 Available Methods

### 1. **Ensemble Voting** (`ensemble`)
- **Source**: `mutil_label_merge.ipynb`
- **Effect**: Strongest improvement
- **What it does**: 
  - Applies ensemble-like voting to segmentation
  - Removes small components
  - Smooths boundaries with morphological operations
- **Best for**: Noisy segmentations with artifacts

### 2. **ROI-based Processing** (`roi`)
- **Source**: `pred_sample.ipynb`
- **Effect**: Focused processing
- **What it does**:
  - Finds organ centers
  - Processes regions of interest around centers
  - Applies morphological operations in ROIs
- **Best for**: Large images with focused organ regions

### 3. **Center Constraints** (`center`)
- **Source**: `get_center_2d.ipynb`, `get_center_3d.ipynb`
- **Effect**: Anatomical consistency
- **What it does**:
  - Calculates organ centers
  - Removes components far from expected centers
  - Enforces anatomical constraints
- **Best for**: Segmentations with anatomical knowledge

### 4. **Multi-label Fusion** (`fusion`)
- **Source**: `mutil_label_merge.ipynb`
- **Effect**: Class-specific processing
- **What it does**:
  - Applies organ-specific constraints
  - Ensures heart connectivity
  - Maintains lung symmetry
  - Enforces aorta constraints
- **Best for**: Multi-organ segmentation with anatomical knowledge

### 5. **Combined Processing** (`combined`)
- **Effect**: Maximum improvement
- **What it does**:
  - Applies all methods sequentially
  - Ensemble → ROI → Center → Fusion
- **Best for**: Maximum quality improvement

## ⚙️ Parameters

### Required Parameters
- `--input_folder`: Input folder containing NIfTI files
- `--output_folder`: Output folder for processed files

### Optional Parameters
- `--num_classes`: Number of classes (default: 5)
- `--method`: Processing method (default: ensemble)
- `--roi_size`: Size of ROI for processing (default: 128)
- `--pattern`: File pattern to match (default: *.nii.gz)

## 🔧 Key Features

### **Ensemble Voting**
- Implements majority voting logic from `mutil_label_merge.ipynb`
- Removes small components
- Applies morphological smoothing
- Best for noisy predictions

### **ROI Processing**
- Center detection from `get_center_2d.ipynb` and `get_center_3d.ipynb`
- ROI-based processing from `pred_sample.ipynb`
- Focused processing around organ centers
- Efficient for large images

### **Center Constraints**
- Anatomical constraint enforcement
- Distance-based filtering
- Organ-specific processing
- Medical knowledge integration

### **Multi-label Fusion**
- Class-specific processing
- Heart connectivity enforcement
- Lung symmetry maintenance
- Aorta constraint application

## 📝 Example Usage

```bash
# Process your validation predictions with ensemble voting
python ensemble_postprocessing.py \
    --input_folder val2/pred \
    --output_folder val2/pred_ensemble \
    --num_classes 5 \
    --method ensemble

# Process with combined methods for maximum quality
python ensemble_postprocessing.py \
    --input_folder val2/pred \
    --output_folder val2/pred_combined_ensemble \
    --num_classes 5 \
    --method combined \
    --roi_size 128

# Then compute metrics
python compute_metrics.py \
    --ref_folder val2/gt \
    --pred_folder val2/pred_combined_ensemble \
    --ref_extension .nii.gz \
    --pred_extension .nii.gz \
    --num_classes 5 \
    --save_folder metrics_ensemble
```

## 🎯 Recommendations

1. **Start with ensemble voting**: Usually gives the best improvement
2. **Try combined for maximum quality**: If you need the best possible results
3. **Use ROI processing for large images**: More efficient processing
4. **Use center constraints for anatomical consistency**: Leverages medical knowledge
5. **Adjust roi_size**: Smaller for fine details, larger for coarse processing
