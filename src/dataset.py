import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class HeightEstimationDataset(Dataset):
    """Dataset for multi-view aerial imagery height estimation.

    Expects data structure:
    - DSM: <clip>_dsm.png
    - Ortho: <clip>_.*_nadir.tiff.jpg
    - Obliques: <clip>_.*_oblique-{north,south,east,west}.tiff.jpg
    """

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        height_norm_params: Optional[Dict[str, float]] = None,
        compute_norm_params: bool = False
    ):
        """
        Args:
            data_dir: Path to data folder containing images
            split: 'train' or 'val' (validation uses clip % 5 == 0)
            height_norm_params: Dict with 'min' and 'max' for height normalization
            compute_norm_params: If True, compute and store normalization params
        """
        self.data_dir = Path(data_dir)
        self.split = split

        # Find all valid clips (those with all 6 files)
        self.clips = self._find_valid_clips()

        # Split train/val
        if split == 'train':
            self.clips = [c for c in self.clips if c['clip_id'] % 5 != 0]
        elif split == 'val':
            self.clips = [c for c in self.clips if c['clip_id'] % 5 == 0]
        else:
            raise ValueError(f"Split must be 'train' or 'val', got {split}")

        print(f"{split.upper()} set: {len(self.clips)} clips")

        # Height normalization
        if compute_norm_params:
            self.height_norm_params = self._compute_height_norm_params()
            print(f"Computed height normalization: min={self.height_norm_params['min']:.2f}, "
                  f"max={self.height_norm_params['max']:.2f}")
        elif height_norm_params is not None:
            self.height_norm_params = height_norm_params
        else:
            # Default normalization (will be overridden when params are computed)
            self.height_norm_params = {'min': 0.0, 'max': 255.0}

    def _find_valid_clips(self) -> List[Dict[str, any]]:
        """Find all clips with complete set of 6 files.

        Optimized version: Uses single os.listdir() instead of multiple glob operations.
        This is much faster on WSL2 and other slow filesystems.
        """
        # Get all files in one operation
        all_files = os.listdir(self.data_dir)

        # Group files by clip_id
        clips_dict = {}

        for filename in all_files:
            # Parse DSM files: <clip>_dsm.png
            dsm_match = re.match(r'^(\d+)_dsm\.png$', filename)
            if dsm_match:
                clip_id = int(dsm_match.group(1))
                if clip_id not in clips_dict:
                    clips_dict[clip_id] = {}
                clips_dict[clip_id]['dsm'] = str(self.data_dir / filename)
                continue

            # Parse ortho nadir: <clip>_.*_nadir.tiff.jpg
            ortho_match = re.match(r'^(\d+)_.*_nadir\.tiff\.jpg$', filename)
            if ortho_match:
                clip_id = int(ortho_match.group(1))
                if clip_id not in clips_dict:
                    clips_dict[clip_id] = {}
                clips_dict[clip_id]['ortho'] = str(self.data_dir / filename)
                continue

            # Parse oblique files: <clip>_.*_oblique-{direction}.tiff.jpg
            oblique_match = re.match(r'^(\d+)_.*_oblique-(north|south|east|west)\.tiff\.jpg$', filename)
            if oblique_match:
                clip_id = int(oblique_match.group(1))
                direction = oblique_match.group(2)
                if clip_id not in clips_dict:
                    clips_dict[clip_id] = {}
                clips_dict[clip_id][f'oblique_{direction}'] = str(self.data_dir / filename)
                continue

        # Build list of valid clips (those with all 6 files)
        valid_clips = []
        required_keys = {'dsm', 'ortho', 'oblique_north', 'oblique_south', 'oblique_east', 'oblique_west'}

        for clip_id, files in clips_dict.items():
            if required_keys.issubset(files.keys()):
                valid_clips.append({
                    'clip_id': clip_id,
                    'dsm': files['dsm'],
                    'ortho': files['ortho'],
                    'oblique_north': files['oblique_north'],
                    'oblique_south': files['oblique_south'],
                    'oblique_east': files['oblique_east'],
                    'oblique_west': files['oblique_west'],
                })

        valid_clips.sort(key=lambda x: x['clip_id'])
        return valid_clips

    def _compute_height_norm_params(self, max_samples: int = 500) -> Dict[str, float]:
        """Compute min/max height values from a sample of training DSMs after grounding.

        Args:
            max_samples: Maximum number of clips to sample for computing normalization.
                        Using a sample is much faster for large datasets (e.g., 5000+ clips).
        """
        import random

        # Sample clips if we have more than max_samples
        if len(self.clips) > max_samples:
            sample_clips = random.sample(self.clips, max_samples)
            print(f"Computing height normalization from {max_samples} sampled clips (out of {len(self.clips)})...")
        else:
            sample_clips = self.clips
            print(f"Computing height normalization from all {len(self.clips)} clips...")

        all_relative_heights = []

        for clip in sample_clips:
            dsm = np.array(Image.open(clip['dsm'])).astype(np.float32)

            # Ground each DSM: clip at 2nd percentile and subtract to set base to 0m
            percentile_2 = np.percentile(dsm, 2)
            dsm_grounded = np.maximum(dsm - percentile_2, 0)

            all_relative_heights.append(dsm_grounded.flatten())

        all_relative_heights = np.concatenate(all_relative_heights)
        return {
            'min': 0.0,  # Always 0 after grounding
            'max': float(np.percentile(all_relative_heights, 99))  # Use 99th percentile for robustness
        }

    def _load_and_preprocess_rgb(self, path: str, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
        """Load RGB image and preprocess to tensor."""
        img = Image.open(path).convert('RGB')
        img = img.resize(size, Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1)  # HWC -> CHW
        return img

    def _load_and_preprocess_dsm(self, path: str, size: Tuple[int, int] = (32, 32)) -> torch.Tensor:
        """Load DSM and preprocess to tensor.

        Steps:
        1. Load raw DSM
        2. Ground to 0m by subtracting 2nd percentile (removes absolute elevation offset)
        3. Resize to target size
        4. Normalize to [0, 1] using dataset statistics
        """
        dsm = Image.open(path)
        dsm = np.array(dsm).astype(np.float32)

        # Ground the DSM: clip at 2nd percentile and subtract to set base to 0m
        # This removes absolute elevation offset (e.g., 1000m vs 4m base elevation)
        percentile_2 = np.percentile(dsm, 2)
        dsm = np.maximum(dsm - percentile_2, 0)

        # Resize after grounding to preserve relative heights
        dsm_img = Image.fromarray(dsm)
        dsm_img = dsm_img.resize(size, Image.BILINEAR)
        dsm = np.array(dsm_img).astype(np.float32)

        # Normalize to [0, 1] using dataset-wide statistics
        h_max = self.height_norm_params['max']
        dsm = dsm / (h_max + 1e-8)  # min is always 0 after grounding
        dsm = np.clip(dsm, 0, 1)  # Clip outliers beyond 99th percentile

        dsm = torch.from_numpy(dsm).unsqueeze(0)  # Add channel dimension
        return dsm

    def __len__(self) -> int:
        return len(self.clips)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Returns:
            images: Dict with keys 'ortho', 'north', 'south', 'east', 'west'
                    Each value is a (3, 256, 256) tensor
            target: (1, 32, 32) tensor with normalized heights
        """
        clip = self.clips[idx]

        # Load all images
        images = {
            'ortho': self._load_and_preprocess_rgb(clip['ortho']),
            'north': self._load_and_preprocess_rgb(clip['oblique_north']),
            'south': self._load_and_preprocess_rgb(clip['oblique_south']),
            'east': self._load_and_preprocess_rgb(clip['oblique_east']),
            'west': self._load_and_preprocess_rgb(clip['oblique_west']),
        }

        # Load target DSM
        target = self._load_and_preprocess_dsm(clip['dsm'])

        return images, target

    def denormalize_height(self, normalized_height: torch.Tensor) -> torch.Tensor:
        """Convert normalized height back to relative height scale (meters from base).

        Note: Returns relative heights (grounded to 0m), not absolute elevations.
        """
        h_max = self.height_norm_params['max']
        return normalized_height * h_max


def create_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    force_recompute: bool = False
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, Dict[str, float]]:
    """Create train and validation dataloaders.

    Args:
        data_dir: Path to data directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes for data loading
        force_recompute: If True, recompute normalization params even if cached

    Returns:
        train_loader, val_loader, height_norm_params
    """
    import json

    data_path = Path(data_dir)
    cache_file = data_path / 'height_norm_params_cache.json'

    # Try to load cached normalization params
    if cache_file.exists() and not force_recompute:
        print(f"Loading cached normalization params from {cache_file}")
        with open(cache_file, 'r') as f:
            height_norm_params = json.load(f)
        print(f"  min={height_norm_params['min']:.2f}, max={height_norm_params['max']:.2f}")

        # Create datasets with cached params
        train_dataset = HeightEstimationDataset(
            data_dir=data_dir,
            split='train',
            height_norm_params=height_norm_params
        )
    else:
        # Compute normalization params
        if force_recompute:
            print("Force recomputing normalization params...")
        else:
            print("No cached params found. Computing normalization params...")

        train_dataset = HeightEstimationDataset(
            data_dir=data_dir,
            split='train',
            compute_norm_params=True
        )

        # Get and cache normalization params
        height_norm_params = train_dataset.height_norm_params

        # Save to cache
        with open(cache_file, 'w') as f:
            json.dump(height_norm_params, f, indent=2)
        print(f"Saved normalization params to {cache_file}")

    # Create val dataset with same normalization
    val_dataset = HeightEstimationDataset(
        data_dir=data_dir,
        split='val',
        height_norm_params=height_norm_params
    )

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, height_norm_params
