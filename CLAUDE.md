# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Height estimation machine learning model for aerial imagery using PyTorch. The model takes 5 input images (ortho overhead, north/east/south/west obliques) and outputs a Digital Surface Model (DSM) raster indicating height values.

## Architecture

**Multi-branch encoder with cross-attention fusion:**
- 5 ResNet50 backbones (ImageNet pre-trained) - one per input image
- Design: 256×256 input → 64×64 output (4× downsampling)
- Cross-attention fusion using ortho as anchor for querying oblique features
- U-Net style decoder with skip connections and sub-pixel convolution upsampling
- **Emphasis**: Rich multi-view context (256 channels) over skip details (64 channels), 4:1 ratio

**Detailed flow:**
1. **Encoder (per image - 5× parallel):**
   - Input: 256×256×3 RGB (normalized to [0,1])
   - ResNet50 through layer2: 32×32×512 (saved for skip, ortho only)
   - ResNet50 layer3: 16×16×1024
   - 1×1 conv reduce to: 16×16×256
   - Strided conv (stride=2) downsample to: 8×8×256
   - Output: 5×(batch, 8, 8, 256)

2. **Fusion via cross-attention:**
   - Flatten spatial: 5×(batch, 64, 256) where 64 = 8×8 positions
   - Total: 320 spatial positions (5 views × 64)
   - Cross-attention: ortho as queries (B, 64, 256), all 5 views as keys/values (B, 320, 256)
   - Output: (batch, 64, 256) fused features
   - Re-spatialize to: (batch, 8, 8, 256)

3. **Skip connection (ortho only):**
   - Ortho ResNet50 layer2: 32×32×512
   - 1×1 conv reduce to: 32×32×64 (preserved at full resolution)

4. **Decoder:**
   - Upsample to (batch, 16, 16, 256) via sub-pixel convolution
   - Upsample to (batch, 32, 32, 256) via sub-pixel convolution
   - Concatenate with skip: (batch, 32, 32, 320) = 256 context + 64 skip
   - Conv block: (batch, 32, 32, 128)
   - Upsample to (batch, 64, 64, 64) via sub-pixel convolution
   - Final 3×3 conv to (batch, 64, 64, 1) height output

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
- Resample targets to 64×64×1 using bilinear interpolation
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

**Memory and performance:**
- Resolution: 256×256 input → 64×64 output (4× downsampling)
- Expected memory usage: ~6-8GB VRAM with batch_size=8
- If OOM errors occur: reduce batch size to 4 or 2
- **Fast dataloader**: Optimized for WSL2/slow filesystems
  - Uses single `os.listdir()` instead of ~1000 glob operations
  - Startup time: <1 second for 5000 clips vs 15+ minutes with glob
  - Critical optimization for development workflow

**Model specifications:**
- Total parameters: ~50M
- Context channels at fusion: 256 (rich multi-view learning)
- Skip channels: 64 (lightweight spatial refinement)
- Cross-attention operates on 320 positions (5 views × 64 spatial)

**Key implementation priorities:**
1. Data loader with optimized file matching (single listdir) and validation split
2. Model architecture with pre-trained ResNet50 backbones
3. Cross-attention fusion module (256-dimensional features)
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
