# Multimodal Clinical Diagnosis System using Custom Deep Learning Architectures

This project implements a high-accuracy chest X-ray disease classifier using ResNet50 with Focal Loss for handling class imbalance.

## Overview

**Key Optimizations for Accuracy & Large Datasets:**
- **Backbone**: ResNet50 pretrained on ImageNet (10M+ parameters vs. 462K for custom CNN)
- **Loss Function**: Focal Loss with γ=2.0 for handling severe class imbalance in multi-label medical imaging
- **Batch Size**: Reduced to 8 (vs. 16) for better gradient signals on large datasets
- **Augmentation**: Aggressive augmentation including rotation (15°), affine transforms, elastic deformation, Gaussian blur
- **Threshold Optimization**: Per-class decision thresholds optimized on validation set to maximize F1 scores
- **Learning Rate Scheduling**: ReduceLROnPlateau to dynamically adjust learning rate
- **Early Stopping**: Stops training when validation loss plateaus after 10 epochs of no improvement
- **Epochs**: Extended to 100 (vs. 20) to ensure convergence with reduced batch size

## Chest X-ray Dataset & Setup

- **Dataset**: NIH ChestXray14 (14 disease classes, 112K images)
- **Location**: Extract to `~/Downloads/archive/`
- **Format**: CSV metadata (`Data_Entry_2017.csv`) + nested `images_###/images/` folders
- **Testing**: Use `--max-images 500` for validation, or `--max-images 0` for full dataset training

## Run Training

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Quick test (500 images, 10 epochs):**
   ```bash
   python train_chest_xray.py --data-root ~/Downloads/archive --max-images 500 --epochs 10
   ```

3. **Full training (all images, 100 epochs with Focal Loss):**
   ```bash
   python train_chest_xray.py --data-root ~/Downloads/archive --max-images 0 --epochs 100 --batch-size 8
   ```

4. **Resume inference on new images:**
   ```bash
   python train_chest_xray.py --inference-only --checkpoint outputs/chest_xray_cnn.pt --external-test-folder input/
   ```

## Command-Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data-root` | `~/Downloads/archive` | Path to NIH ChestXray14 dataset |
| `--max-images` | 0 | Max images to use (0 = full dataset) |
| `--batch-size` | 8 | Batch size (reduced for better gradients on large datasets) |
| `--epochs` | 100 | Number of training epochs |
| `--learning-rate` | 1e-4 | Adam learning rate (lower for pretrained ResNet50) |
| `--image-size` | 224 | Image resolution (224×224) |
| `--output-dir` | `outputs/` | Directory for checkpoints and metrics |
| `--seed` | 42 | Random seed for reproducibility |
| `--use-focal-loss` | True | Use Focal Loss for class imbalance |

## Output Files

```
outputs/
├── chest_xray_cnn.pt              # Final trained model checkpoint
├── chest_xray_cnn_best.pt         # Best checkpoint (lowest validation loss)
├── chest_xray_classes.json        # Class names metadata
├── test_metrics.json              # Per-class and aggregate metrics + optimized thresholds
├── test_previews/                 # Grad-CAM visualizations for first 5 test samples
└── external_test_previews/        # (optional) Inference results on external images
```

## Model Architecture

**ResNet50 Backbone** (pretrained on ImageNet):
- 4 residual blocks with skip connections
- ~25M parameters

**Custom Classifier Head**:
- Dropout(0.5) → Linear(2048 → 512) → ReLU → BatchNorm
- Dropout(0.3) → Linear(512 → 256) → ReLU → BatchNorm
- Dropout(0.2) → Linear(256 → 14 diseases)

**Total Parameters**: ~25.5M

## Loss Function: Focal Loss

Addresses class imbalance in medical imaging by down-weighting easy negatives:

```
FL(p_t) = -α(1 - p_t)^γ log(p_t)
```

- **α = 0.25**: Weighting for positive/negative class
- **γ = 2.0**: Focusing parameter (higher = more focus on hard negatives)

## Expected Performance

With full NIH ChestXray14 dataset:
- **Micro F1**: 0.7-0.8 (all labels aggregated)
- **Macro F1**: 0.5-0.7 (per-disease average, handles imbalance)
- **Per-disease F1**: 0.3-0.9 (varies by disease prevalence)

With 500-1000 images (fast validation):
- **Micro F1**: 0.6-0.7
- **Macro F1**: 0.4-0.6

## Data Augmentation Strategy

**Training augmentation** (applied during training):
- Random rotation: ±15°
- Affine transform: ±5% translation, 90-110% scale, ±10° shear
- Horizontal flip: 50% probability
- Color jitter: ±20% brightness/contrast/saturation
- Gaussian blur: σ ∈ [0.1, 2.0]
- Random erasing: 20% of images, removes 2-15% of area
- Random invert: 10% probability (medical imaging robustness)

**Validation/Test augmentation** (no randomness):
- Grayscale → RGB conversion
- Resize to 224×224
- Normalization (mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

## Per-Class Threshold Optimization

After training, thresholds are optimized on the validation set:
1. For each disease class, grid search thresholds [0.1, 0.95]
2. Select threshold that maximizes F1 score
3. Apply per-class thresholds during test evaluation
4. Store in `test_metrics.json` for reproducibility

Example optimized thresholds:
```json
{
  "Infiltration": 0.45,
  "Pneumonia": 0.60,
  "Pneumothorax": 0.55,
  ...
}
```

## Grad-CAM Visualization

Each test preview shows:
- **Left**: Original chest X-ray image
- **Right**: Grad-CAM heatmap highlighting regions driving the top prediction
- **Annotations**: Predicted diseases, confidence scores, medical interpretation

## Troubleshooting

### Out-of-memory errors
- Reduce `--batch-size` to 4 or 2
- Reduce `--image-size` to 192 or 160
- Use `--max-images` to train on a subset

### Poor accuracy on training
- Ensure dataset path is correct: `~/Downloads/archive/Data_Entry_2017.csv`
- Try `--max-images 1000` first to verify pipeline
- Check that images folder structure matches: `images_001/images/*.png`

### Training too slow
- Enable GPU: ensure CUDA is installed (`torch.cuda.is_available()`)
- Increase `--batch-size` if memory allows
- Use `--max-images 5000` for faster validation cycles

## Citation

If you use this code, please cite:

- **NIH ChestXray14**: Wang et al. (2017) "ChestX-ray8: Hospital-scale Chest X-ray Database"
- **Focal Loss**: Lin et al. (2017) "Focal Loss for Dense Object Detection"
- **ResNet**: He et al. (2015) "Deep Residual Learning for Image Recognition"
