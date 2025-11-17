# Training and Validation Metrics

This document explains all metrics logged during training and validation, including their units and how to interpret them.

## Important: Units and Normalization

**All error metrics are computed in NORMALIZED HEIGHT SPACE [0, 1], not meters.**

To convert to meters, multiply by the `max` parameter from `checkpoints/height_norm_params.json` (typically the 99th percentile of grounded heights in the training set).

---

## Training Metrics

### 1. **train_loss** (Logged to CSV)
- **Description**: Combined weighted loss function
- **Components**:
  - Huber loss (weight=1.0)
  - Gradient loss (weight=0.1)
  - Multi-scale loss (weight=0.15)
  - Edge-aware smoothness (weight=0.05)
- **Units**: Normalized height space [0, 1]
- **Purpose**: Primary optimization objective

### Loss Components (shown in progress bar):

#### **huber**
- **Description**: Smooth L1 loss (beta=1.0)
- **Behavior**: Quadratic for small errors (|error| < 1.0), linear for large errors
- **Units**: Normalized [0, 1]
- **Purpose**: Robust primary loss, less sensitive to outliers than L2

#### **grad** (gradient loss)
- **Description**: L1 loss on Sobel gradients (x and y directions)
- **Units**: Normalized gradient magnitude
- **Purpose**: Preserves sharp edges and boundaries (e.g., building edges)
- **Computed as**: `|∇pred - ∇target|` using Sobel filters

#### **smooth** (edge-aware smoothness)
- **Description**: Penalizes height variation in smooth RGB regions
- **Units**: Weighted normalized gradient magnitude
- **Purpose**: Reduces noise in flat areas while preserving edges
- **Mechanism**: Weight = exp(-|∇RGB|), so high RGB gradient → low penalty

*(Multi-scale loss is computed but not shown in progress bar)*

---

## Validation Metrics

### 2. **val_loss** (Logged to CSV)
- **Description**: L1 loss on validation set (NOT combined loss)
- **Formula**: `mean(|prediction - target|)`
- **Units**: Normalized height [0, 1]
- **Purpose**: Model checkpoint selection (best model saved)
- **To convert to meters**: Multiply by `height_norm_params['max']`

### 3. **val_rmse** (Logged to CSV)
- **Description**: Root Mean Squared Error
- **Formula**: `sqrt(mean((prediction - target)²))`
- **Units**: Normalized height [0, 1]
- **Purpose**: Emphasizes larger errors (quadratic penalty)
- **To convert to meters**: Multiply by `height_norm_params['max']`

### 4. **val_median_ae** (Logged to CSV)
- **Description**: Median Absolute Error
- **Formula**: `median(|prediction - target|)`
- **Units**: Normalized height [0, 1]
- **Purpose**: Robust central tendency metric (not affected by outliers)
- **To convert to meters**: Multiply by `height_norm_params['max']`

*(Note: L1 loss equals Mean Absolute Error, so val_loss also serves as val_mae)*

---

## Additional Logged Metrics

### 5. **lr** (Learning rate)
- **Description**: Current learning rate
- **Units**: Dimensionless (typically 1e-6 to 1e-4)
- **Schedule**: Linear warmup (10 epochs) + cosine annealing

### 6. **time_epoch**
- **Description**: Wall-clock time per epoch
- **Units**: Seconds
- **Purpose**: Training efficiency monitoring

---

## Height Normalization Details

**Critical preprocessing** (from `src/dataset.py:175-203`):

1. **Grounding**: Each DSM is grounded by subtracting its 2nd percentile
   - Purpose: Remove absolute elevation offset (e.g., 1000m mountain base vs 4m coastal base)
   - Result: All DSMs start at ~0m relative height

2. **Normalization**: Divide by dataset-wide 99th percentile
   - `height_norm_params['min']` = 0.0 (always, after grounding)
   - `height_norm_params['max']` = 99th percentile of all grounded heights
   - Heights clipped to [0, 1]

3. **Denormalization**: `height_meters = normalized_height × max`

---

## Example Interpretation

If you see:
```
val_loss: 0.0234
val_rmse: 0.0456
val_median_ae: 0.0189
```

And `height_norm_params['max']` = 30.0 meters, this means:
- **Mean Absolute Error**: 0.0234 × 30 = **0.70 meters**
- **RMSE**: 0.0456 × 30 = **1.37 meters**
- **Median Absolute Error**: 0.0189 × 30 = **0.57 meters**

---

## Key Files

- **Metrics computation**: `src/utils.py:27-53`
- **Loss functions**: `src/utils.py:56-231`
- **Height normalization**: `src/dataset.py:175-203`
- **Normalization params saved to**: `checkpoints/height_norm_params.json`
- **Training logs**: `training_log.csv`

---

## CSV Log Format

The `training_log.csv` file contains the following columns:

| Column | Description | Units |
|--------|-------------|-------|
| epoch | Epoch number | Integer |
| train_loss | Combined training loss | Normalized [0,1] |
| val_loss | Validation L1 loss (MAE) | Normalized [0,1] |
| val_rmse | Validation RMSE | Normalized [0,1] |
| val_median_ae | Validation median absolute error | Normalized [0,1] |
| lr | Learning rate | Dimensionless (1e-6 to 1e-4) |
| time_epoch | Time per epoch | Seconds |
