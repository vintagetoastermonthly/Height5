# Height Estimation Model

Multi-view aerial imagery height estimation using PyTorch. Predicts Digital Surface Models (DSM) from 5 input views: ortho overhead + 4 oblique directions.

## Project Structure

```
depth5/
├── data/                          # Place your data here
│   ├── <clip>_dsm.png            # Target DSM rasters
│   ├── <clip>_.*_nadir.tiff.jpg  # Ortho images
│   └── <clip>_.*_oblique-{north,south,east,west}.tiff.jpg
├── src/
│   ├── dataset.py                # Data loading and preprocessing
│   ├── model.py                  # Model architecture
│   ├── train.py                  # Training script
│   ├── utils.py                  # Utilities and visualization
│   └── inference.py              # Inference script
├── checkpoints/                   # Saved models
├── examples/                      # Visualization outputs
├── requirements.txt
├── CLAUDE.md                      # Project documentation
└── proposal.md                    # Technical specification
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run tests to verify installation:
```bash
pytest tests/ -v
```

3. Prepare your data:

   **Option A: Use dummy data for testing**
   ```bash
   python generate_dummy_data.py --data_dir data --num_clips 20
   ```

   **Option B: Use your own data**
   - Place all images in the `data/` folder
   - Ensure all clips have 6 files: 1 DSM + 1 ortho + 4 obliques
   - Validation split: clips where `clip_id % 5 == 0`

## Training

**⚡ Performance Note:** For large datasets (1000+ clips), the first run computes normalization from a sample of 500 clips (~1-2 min) and caches it. Subsequent runs use the cache and start immediately. See [PERFORMANCE.md](PERFORMANCE.md) for details.

Basic training:
```bash
python src/train.py --data_dir data --batch_size 8
```

With custom parameters:
```bash
python src/train.py \
  --data_dir data \
  --batch_size 4 \
  --epochs 200 \
  --lr 1e-4 \
  --warmup_epochs 10 \
  --patience 30
```

Key arguments:
- `--batch_size`: Start with 8, reduce to 4 or 2 if OOM (default: 8)
- `--epochs`: Maximum training epochs (default: 200)
- `--lr`: Initial learning rate (default: 1e-4)
- `--warmup_epochs`: Linear warmup period (default: 10)
- `--patience`: Early stopping patience (default: 30)
- `--checkpoint_dir`: Where to save models (default: checkpoints)
- `--examples_dir`: Where to save visualizations (default: examples)

## Monitoring Training

Training progress is logged to `training_log.csv` with:
- Epoch number
- Train/validation loss
- Validation RMSE and median absolute error
- Learning rate
- Time per epoch

Best model visualizations are saved to `examples/` after each improvement.

## Model Architecture

Multi-view height estimation with **256×256 input → 64×64 output** using cross-attention fusion.

### Architecture Overview

The model emphasizes **rich multi-view context learning (256 channels)** over skip connection details (64 channels), with a 4:1 ratio at concatenation.

### Detailed Flow

#### 1. Encoder: Processing Each View (5× in parallel)

Each of the 5 views (ortho + 4 obliques) goes through **identical ResNet50 encoders**:

```
256×256×3 RGB input
   ↓ ResNet50 conv1 + bn + relu + maxpool
64×64×64
   ↓ ResNet50 layer1
64×64×256
   ↓ ResNet50 layer2 (stride=2)
32×32×512  ← SAVED for skip connection (ortho only)
   ↓ ResNet50 layer3 (stride=2)
16×16×1024
   ↓ 1×1 conv: reduce channels (1024→256)
16×16×256
   ↓ 3×3 conv: spatial downsample (stride=2)
8×8×256  ← READY FOR FUSION
```

**Result:** All 5 views at **8×8×256** with equal representation.

#### 2. Cross-Attention Fusion: Combining Multi-View Information

**Input:** 5 feature maps (B, 256, 8, 8) - ortho, north, south, east, west

**Fusion mechanism:**
1. Flatten each view: (B, 256, 8, 8) → (B, 64, 256) where 64 = 8×8 positions
2. Total: 5 views × 64 positions = **320 spatial positions**
3. Cross-attention setup:
   - **Queries (Q):** Ortho features (B, 64, 256)
   - **Keys/Values (K, V):** All views concatenated (B, 320, 256)
4. For each of 64 ortho positions, attention weights determine relevant information across all 320 positions
5. Output: Fused features (B, 256, 8, 8)

**Key insight:** Each output position "attends" to all 5 views, weighting information by relevance. If the north oblique clearly shows a building edge, attention weights it higher for those positions.

#### 3. Skip Connection: High-Resolution Detail from Ortho

```
Ortho layer2: 32×32×512 (from ResNet50 encoder)
   ↓ 1×1 conv: reduce channels (512→64)
32×32×64  ← SKIP CONNECTION
```

Preserved at full 32×32 resolution (1,024 spatial positions).

#### 4. Decoder: Upsampling with Skip Integration

```
8×8×256 (fused multi-view context)
   ↓ SubPixelUpsample (2×)
16×16×256 (maintaining rich context)
   ↓ SubPixelUpsample (2×)
32×32×256 (maintaining rich context)
   ↓ Concatenate with skip
32×32×320 (256 context + 64 skip = 4:1 emphasis)
   ↓ Conv block (2× conv3×3 + BN + ReLU)
32×32×128
   ↓ SubPixelUpsample (2×)
64×64×64
   ↓ Final conv3×3
64×64×1 (height map output)
```

**Design rationale:** The upsampled features (8×8→32×32) contain semantic understanding from multi-view fusion, while skip connection provides fine-grained spatial details. Concatenating at 32×32 allows the decoder to:
- Use semantic info to understand "what" is present (building, tree, ground)
- Use skip details to precisely locate "where" edges and boundaries are
- Upsample once more (32×32→64×64) with combined information

### Architecture Specifications

- **Total parameters**: ~50M
- **Memory**: ~6-8GB VRAM at batch_size=8
- **Context channels**: 256 (emphasizes multi-view learning)
- **Skip channels**: 64 (lightweight spatial refinement)
- **Fusion resolution**: 8×8 with 64 spatial positions
- **Output resolution**: 64×64 (4× more detail than initial 32×32 design)

## Inference

Use trained model for prediction:
```bash
python src/inference.py \
  --checkpoint checkpoints/best_model.pth \
  --data_dir data \
  --output_dir predictions
```

## Testing

Run all tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=term-missing --cov-report=html
```

Or use the provided script:
```bash
./run_tests.sh
```

Test coverage includes:
- Model architecture and forward pass (49 tests total)
- Data loading and preprocessing
- Metrics computation
- Learning rate scheduling
- Visualization utilities

## Dummy Data Generation

Generate synthetic data for testing:
```bash
python generate_dummy_data.py --data_dir data --num_clips 20 --seed 42
```

This creates 20 property clips (120 files total) with:
- Random noise DSM images (16-bit PNG)
- Random RGB images for ortho and obliques
- Varied dimensions (400-800 pixels)
- 16 training clips + 4 validation clips

## Notes

- Resolution: 256×256 input → 64×64 output (4× downsampling)
- **DSM grounding**: Each DSM is grounded to 0m by subtracting its 2nd percentile
  - Removes absolute elevation offset (e.g., sea level vs mountain scenes)
  - Model learns relative heights (buildings, trees, terrain) not absolute elevations
  - Critical for handling mixed datasets with varied base elevations
- Combined loss: Huber (primary) + gradient + multi-scale + edge-aware smoothness
- Cosine annealing LR schedule with linear warmup
- Early stopping prevents overfitting
- Augmentation: Rotation (90° increments) + image dropout (10% per view)
- ~50M parameters (5 partial ResNet50 encoders + cross-attention + decoder)
- Fast dataloader: optimized for WSL2 with single `os.listdir()` vs multiple glob operations
