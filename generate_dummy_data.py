#!/usr/bin/env python3
"""Generate dummy data for testing the height estimation model."""

import os
import numpy as np
from PIL import Image
from pathlib import Path


def generate_dummy_data(data_dir: str, num_clips: int = 20, seed: int = 42):
    """
    Generate dummy data with random noise images.

    Args:
        data_dir: Directory to create data in
        num_clips: Number of property clips to generate
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    data_path = Path(data_dir)
    data_path.mkdir(exist_ok=True, parents=True)

    print(f"Generating {num_clips} dummy clips in {data_dir}/")

    for clip_id in range(num_clips):
        # Vary image dimensions slightly for realism
        base_size = np.random.randint(400, 800)

        # 1. Generate DSM (16-bit grayscale with height values)
        dsm_size = (base_size, base_size)

        # Vary absolute elevation offset to test grounding (sea level vs mountains)
        # Some clips near sea level (~5m), some in mountains (~1000m)
        base_elevation = np.random.choice([5, 50, 200, 500, 1000, 1500])

        # Relative height variation within scene (0-100m buildings/trees/terrain)
        relative_heights = np.random.randint(0, 100, size=dsm_size, dtype=np.uint16)

        # Add some smooth gradients to make terrain more realistic
        x = np.linspace(0, 1, dsm_size[1])
        y = np.linspace(0, 1, dsm_size[0])
        xx, yy = np.meshgrid(x, y)
        gradient = ((np.sin(xx * 4 * np.pi) + np.cos(yy * 3 * np.pi)) * 15 + 30).astype(np.uint16)

        # Combine: absolute elevation + relative heights + terrain
        dsm = np.clip(base_elevation + relative_heights // 2 + gradient, 0, 65535).astype(np.uint16)

        dsm_img = Image.fromarray(dsm, mode='I;16')
        dsm_path = data_path / f'{clip_id}_dsm.png'
        dsm_img.save(dsm_path)

        # 2. Generate Ortho RGB image
        ortho_size = (base_size, base_size, 3)
        ortho = np.random.randint(50, 200, size=ortho_size, dtype=np.uint8)
        # Add some texture
        ortho = np.clip(ortho + np.random.randn(*ortho_size) * 20, 0, 255).astype(np.uint8)

        ortho_img = Image.fromarray(ortho, mode='RGB')
        # Use random identifier in filename
        identifier = f'prop{clip_id:04d}'
        ortho_path = data_path / f'{clip_id}_{identifier}_nadir.tiff.jpg'
        ortho_img.save(ortho_path, 'JPEG', quality=95)

        # 3. Generate 4 oblique RGB images
        for direction in ['north', 'south', 'east', 'west']:
            oblique_size = (base_size, base_size, 3)
            # Vary color slightly for each direction
            offset = {'north': 0, 'south': 20, 'east': 40, 'west': 60}[direction]
            oblique = np.random.randint(40 + offset, 180 + offset, size=oblique_size, dtype=np.uint8)
            oblique = np.clip(oblique + np.random.randn(*oblique_size) * 25, 0, 255).astype(np.uint8)

            oblique_img = Image.fromarray(oblique, mode='RGB')
            oblique_path = data_path / f'{clip_id}_{identifier}_oblique-{direction}.tiff.jpg'
            oblique_img.save(oblique_path, 'JPEG', quality=95)

        if (clip_id + 1) % 5 == 0:
            print(f"  Generated {clip_id + 1}/{num_clips} clips...")

    print(f"\nDone! Generated {num_clips * 6} files total.")
    print(f"\nDataset split:")
    print(f"  Training clips: {[i for i in range(num_clips) if i % 5 != 0]}")
    print(f"  Validation clips: {[i for i in range(num_clips) if i % 5 == 0]}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate dummy data for height estimation')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory to create data in (default: data)')
    parser.add_argument('--num_clips', type=int, default=20,
                        help='Number of property clips to generate (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    args = parser.parse_args()

    generate_dummy_data(args.data_dir, args.num_clips, args.seed)
