# Height estimation model
## Overview
* height estimation machine learning model similar concept to a depth estimation model.
* for aerial imagery
* 5 input images: ortho overhead, north oblique, east oblique, south oblique, west oblique
* input images are RGB.
* 1 target: DSM raster aligned with the ortho overhead image.
* output raster is single band with pixel value indicating height.
* images have variable dimensions.

## Model design
* PyTorch ResNet50 5x backbones (pre-trained on ImageNet)
* Loosely modelled on DeepLabV3+ with 5 input images processed through encoders, then fused with self-attention
* Memory constraint: 256x256 input → 32x32 output (8x downsampling)

**Encoder path:**
* Standardize all image shapes to 256x256 on input
* Pass each through ResNet50, extract features from layer3 (before final pooling)
* ResNet50 layer3 output: 32x32x1024 features
* Apply 1x1 conv to reduce to 32x32x64 per image
* Downsample to 8x8x64 using strided conv (for memory efficiency)
* Output shape: 5x(b,8,8,64)

**Fusion via self-attention:**
* Flatten spatial dims: 5x(b,64,64) where 64 features × 64 spatial positions
* Apply cross-attention: ortho features as queries, all 5 views (including ortho) as keys/values
* This allows ortho to selectively attend to relevant oblique information
* Output: (b,64,64) fused features
* Re-spatialize to (b,8,8,64)

**Decoder path:**
* Upsample (b,8,8,64) → (b,16,16,128) using sub-pixel convolution
* Extract skip connection from ortho ResNet50 layer2: (b,32,32,512) → reduce to (b,32,32,64) with 1x1 conv
* Downsample skip to (b,16,16,64)
* Concatenate with decoder features: (b,16,16,192)
* Conv block with residual connection to process combined features
* Upsample (b,16,16,128) → (b,32,32,64) using sub-pixel convolution
* Final 3x3 conv → (b,32,32,1) height output

## Data loading
* Images are in the same folder including the DSM. We'll call the folder "data" and assume it's a subfolder of the project directory.
* Images start with a property identifier number <clip>_.*
* DSM images have pattern <clip>_dsm.png
* Ortho images have pattern <clip>_.*_nadir.tiff.jpg
* Oblique images have pattern <clip>_.*_oblique-north.tiff.jpg, with north, south, east, west directions.
* Only load images for which all 6 rasters are present

**Preprocessing:**
* Resample all inputs to 256x256x3 (RGB)
* Normalize RGB to [0, 1] range (divide by 255)
* DSM format: Assume 16-bit PNG or verify bit depth; if 8-bit, scale appropriately
* **DSM grounding (critical):** Each DSM is grounded to 0m by subtracting its 2nd percentile
  - Removes absolute elevation offset (sea level vs mountain base elevations)
  - Model learns relative heights (buildings, trees, terrain) not absolute elevations
  - Avoids signal contamination from mixed elevation datasets
* Normalize grounded DSM heights: use 99th percentile of relative heights across dataset
* Store normalization parameters for denormalization during inference
* Resample target DSM to 32x32x1 using bilinear interpolation
* Valid and training are in the same folder, so split using valid where clip % 5 == 0, for consistent validation set.
* No augmentation to start (to maintain consistency across 5 aligned views)

## Training

**Loss function:**
* Primary: L1 loss (MAE) between predicted and target height maps
* L1 is robust to outliers and produces reasonable results at low resolution
* Optional enhancement: Add 0.1 × gradient loss for better edge preservation (can add later if needed)

**Optimizer and learning rate:**
* Optimizer: AdamW with weight_decay=0.01
* Initial learning rate: 1e-4
* Learning rate schedule: Cosine annealing with linear warmup
  - Warmup: 10 epochs linear increase from 1e-6 to 1e-4
  - Cosine decay: from 1e-4 to 1e-6 over remaining epochs
* Batch size: Start with 8, reduce to 4 or 2 if OOM (given 5× ResNet50 memory requirements)

**Training procedure:**
* Train for up to 200 epochs
* Save best model during training based on validation L1 loss
* Early stopping: patience of 30 epochs (stop if no validation improvement)
* Save 5 consistent example samples as 7-image strips (5 inputs + target + prediction) to `examples/` folder after each new best epoch
* Output stats after each epoch to `training_log.csv` including:
  - Epoch number, train loss, validation loss
  - Validation RMSE, median absolute error
  - Learning rate, time per epoch

**Evaluation metrics:**
* Primary: Validation L1 (MAE)
* Secondary: RMSE, median absolute error
* Monitor for overfitting via train/val loss divergence
