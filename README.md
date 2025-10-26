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

- **Encoder**: 5× ResNet50 (ImageNet pretrained)
  - Input: 256×256×3 RGB per view
  - Features: 4×4×64 per view

- **Fusion**: Cross-attention
  - Ortho queries oblique features
  - Output: 4×4×64 fused features

- **Decoder**: U-Net style with skip connections
  - Skip from ortho layer2: 16×16×64
  - Output: 32×32×1 height map

- **Total parameters**: ~44M (5 partial ResNet50 encoders)
- **Memory**: ~6-8GB VRAM at batch_size=4

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

- Resolution limited to 256→32 for memory constraints
- Output at 32×32 is 8× downsampling from input
- **DSM grounding**: Each DSM is grounded to 0m by subtracting its 2nd percentile
  - Removes absolute elevation offset (e.g., sea level vs mountain scenes)
  - Model learns relative heights (buildings, trees, terrain) not absolute elevations
  - Critical for handling mixed datasets with varied base elevations
- L1 loss used (robust to outliers at low resolution)
- Cosine annealing LR schedule with linear warmup
- Early stopping prevents overfitting
- No augmentation (maintains 5-view alignment)
- ~44M parameters (5 partial ResNet50 encoders up to layer3)
