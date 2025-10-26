# DSM Grounding Preprocessing

## Problem

When working with DSMs (Digital Surface Models) from different locations, the absolute elevation varies significantly:
- Coastal areas: ~4-50m absolute elevation
- Plains/valleys: ~100-500m absolute elevation  
- Mountains: ~1000-2000m+ absolute elevation

Training a model on mixed elevation data without grounding would force the model to learn these large absolute offsets, which:
- Contaminates the signal (model learns "1000" instead of relative structure)
- Makes learning harder (much larger dynamic range)
- Reduces model effectiveness (absolute elevation is not useful for height estimation)

## Solution: DSM Grounding

Each DSM is "grounded" to 0m by subtracting its 2nd percentile value:

```python
percentile_2 = np.percentile(dsm, 2)
dsm_grounded = np.maximum(dsm - percentile_2, 0)
```

### Why 2nd percentile instead of minimum?
- Robust to noise and outliers
- Avoids being affected by a single noisy pixel
- Still effectively grounds the scene to ~0m

### Result
After grounding:
- All scenes start at approximately 0m
- Model learns **relative heights** (buildings, trees, terrain variations)
- Absolute elevation offset is removed

## Example

```
Clip from coastal area:
  Raw: 5m - 85m (80m range, base at 5m)
  Grounded: 0m - 80m (80m range, base at 0m)

Clip from mountains:
  Raw: 1500m - 1580m (80m range, base at 1500m)  
  Grounded: 0m - 80m (80m range, base at 0m)
```

Both scenes now have the same base elevation (0m) and the model can focus on learning the relative height structure.

## Implementation

See `src/dataset.py`:
- `_compute_height_norm_params()`: Computes normalization using grounded heights
- `_load_and_preprocess_dsm()`: Applies grounding during loading
- Normalization uses 99th percentile of grounded heights for robustness

## Testing

The dummy data generator creates DSMs with varied base elevations (5m to 1500m) to test grounding effectiveness.

```bash
python -c "
import numpy as np
from PIL import Image

# Load and ground a DSM
dsm = np.array(Image.open('data/0_dsm.png'))
print(f'Raw: {dsm.min()}m - {dsm.max()}m')

percentile_2 = np.percentile(dsm, 2)
dsm_grounded = np.maximum(dsm - percentile_2, 0)
print(f'Grounded: {dsm_grounded.min()}m - {dsm_grounded.max()}m')
"
```
