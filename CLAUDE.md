# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Height estimation machine learning model for aerial imagery using PyTorch. The model takes 5 input images (ortho overhead, north/east/south/west obliques) and outputs a Digital Surface Model (DSM) raster indicating height values.

## Architecture

**Multi-branch encoder with cross-attention fusion:**
- 5 ResNet50 backbones (ImageNet pre-trained) - one per input image
- Memory-constrained design: 256x256 input → 32x32 output (8x downsampling)
- Cross-attention fusion using ortho as anchor for querying oblique features
- U-Net style decoder with skip connections and sub-pixel convolution upsampling

**Detailed flow:**
1. **Encoder (per image):**
   - Input: 256x256x3 RGB (normalized to [0,1])
   - ResNet50 layer3 features: 32x32x1024
   - 1x1 conv reduce to: 32x32x64
   - Strided conv downsample to: 8x8x64
   - Output: 5×(batch, 8, 8, 64)

2. **Fusion via cross-attention:**
   - Flatten spatial: 5×(batch, 64, 64) where 64 features × 64 positions
   - Cross-attention: ortho as queries, all 5 views as keys/values
   - Output: (batch, 64, 64) fused features
   - Re-spatialize to: (batch, 8, 8, 64)

3. **Decoder:**
   - Upsample to (batch, 16, 16, 128) via sub-pixel convolution
   - Skip connection: ortho ResNet50 layer2 (batch, 32, 32, 512) → 1x1 conv to (batch, 32, 32, 64) → downsample to (batch, 16, 16, 64)
   - Concatenate: (batch, 16, 16, 192)
   - Conv block with residual connections
   - Upsample to (batch, 32, 32, 64) via sub-pixel convolution
   - Final 3x3 conv to (batch, 32, 32, 1) height output

## Data Structure

**Location:** `data/` folder in project directory

**Naming conventions:**
- DSM: `<clip>_dsm.png`
- Ortho: `<clip>_.*_nadir.tiff.jpg`
- Obliques: `<clip>_.*_oblique-{north,south,east,west}.tiff.jpg`

**Preprocessing:**
- Only use clips with all 6 files present
- Resample inputs to 256x256x3, normalize RGB to [0, 1]
- DSM preprocessing (critical for handling varied elevations):
  - **Ground each DSM**: Subtract 2nd percentile to set base to 0m
  - This removes absolute elevation offset (e.g., 1000m vs 4m base)
  - Model learns relative heights within scene, not absolute elevations
  - Verify bit depth (16-bit preferred)
  - Normalize to [0, 1] using 99th percentile of grounded heights
- Store normalization parameters for inference denormalization
- Resample targets to 32x32x1 using bilinear interpolation
- Train/validation split: validation where `clip % 5 == 0`
- No augmentation initially (to maintain 5-view alignment)

## Training Configuration

**Loss and metrics:**
- Primary loss: L1 (MAE) - robust to outliers at low resolution
- Evaluation metrics: L1, RMSE, median absolute error
- Optional future enhancement: Add 0.1 × gradient loss for edge preservation

**Optimizer and schedule:**
- AdamW optimizer, weight_decay=0.01
- Learning rate: 1e-4 initial
- Schedule: Cosine annealing with linear warmup
  - Warmup: 10 epochs (1e-6 → 1e-4)
  - Cosine decay: (1e-4 → 1e-6) over remaining epochs
- Batch size: 8 (reduce to 4 or 2 if OOM due to 5× ResNet50 memory)

**Training procedure:**
- Epochs: up to 200 with early stopping (patience=30)
- Save best model based on validation L1 loss
- Generate 5 sample visualizations (7-image strips: 5 inputs + target + prediction) after each best epoch → `examples/` folder
- Log to `training_log.csv`: epoch, train/val loss, RMSE, median AE, learning rate, time

## Implementation Notes

**Memory constraints:**
- Resolution limited to 256→32 to manage GPU memory with 5× ResNet50 encoders
- Expected memory usage: ~6-8GB VRAM with batch_size=4
- If OOM errors occur: reduce batch size to 2 or use gradient checkpointing

**Key implementation priorities:**
1. Data loader with proper file matching and validation split
2. Model architecture with pre-trained ResNet50 backbones
3. Cross-attention fusion module
4. Training loop with checkpointing and visualization
5. Height normalization/denormalization utilities

## Development Commands

**Setup:**
```bash
pip install -r requirements.txt
```

**Test model architecture (no data needed):**
```bash
cd src
python model.py
```

**Train model:**
```bash
# Basic training with default parameters
python src/train.py --data_dir data --batch_size 8

# Custom training configuration
python src/train.py \
  --data_dir data \
  --batch_size 4 \
  --epochs 200 \
  --lr 1e-4 \
  --warmup_epochs 10 \
  --patience 30 \
  --checkpoint_dir checkpoints \
  --examples_dir examples
```

**Run inference:**
```bash
# Run on validation set
python src/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data \
  --split val \
  --output_dir predictions

# Save visualizations
python src/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data \
  --save_visualizations
```

**Monitor training:**
- Training logs: `training_log.csv`
- Best model visualizations: `examples/example_{0-4}_epoch{N}.png`
- Checkpoints: `checkpoints/best_model.pth`

## Code Organization

- `src/dataset.py`: Dataset class, file matching, train/val split, normalization
- `src/model.py`: Full model architecture (encoder, fusion, decoder)
- `src/train.py`: Training loop, validation, checkpointing, early stopping
- `src/inference.py`: Inference script for trained models
- `src/utils.py`: Metrics, visualization, learning rate scheduler, CSV logging
