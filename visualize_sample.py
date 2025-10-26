#!/usr/bin/env python3
"""Visualize a sample from the dataset."""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path


def visualize_sample(data_dir: str, clip_id: int = 0):
    """Visualize all 6 images for a given clip."""
    data_path = Path(data_dir)

    # Find files for this clip
    dsm_file = list(data_path.glob(f'{clip_id}_dsm.png'))[0]
    ortho_file = list(data_path.glob(f'{clip_id}_*_nadir.tiff.jpg'))[0]
    oblique_files = {
        'north': list(data_path.glob(f'{clip_id}_*_oblique-north.tiff.jpg'))[0],
        'south': list(data_path.glob(f'{clip_id}_*_oblique-south.tiff.jpg'))[0],
        'east': list(data_path.glob(f'{clip_id}_*_oblique-east.tiff.jpg'))[0],
        'west': list(data_path.glob(f'{clip_id}_*_oblique-west.tiff.jpg'))[0],
    }

    # Load images
    dsm = np.array(Image.open(dsm_file))
    ortho = np.array(Image.open(ortho_file))
    obliques = {k: np.array(Image.open(v)) for k, v in oblique_files.items()}

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Clip {clip_id} - Multi-view Aerial Imagery', fontsize=16, fontweight='bold')

    # Plot DSM
    im = axes[0, 0].imshow(dsm, cmap='terrain')
    axes[0, 0].set_title(f'DSM Target\n{dsm.shape}', fontweight='bold')
    axes[0, 0].axis('off')
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, label='Height (arbitrary units)')

    # Plot ortho
    axes[0, 1].imshow(ortho)
    axes[0, 1].set_title(f'Ortho (Nadir)\n{ortho.shape}', fontweight='bold')
    axes[0, 1].axis('off')

    # Plot obliques
    positions = {'north': (0, 2), 'east': (1, 0), 'south': (1, 1), 'west': (1, 2)}
    for direction, (row, col) in positions.items():
        axes[row, col].imshow(obliques[direction])
        axes[row, col].set_title(f'Oblique {direction.capitalize()}\n{obliques[direction].shape}', fontweight='bold')
        axes[row, col].axis('off')

    plt.tight_layout()
    output_path = Path(data_dir).parent / f'sample_clip_{clip_id}_visualization.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f'Saved visualization to: {output_path}')
    plt.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Visualize a sample from the dataset')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the data (default: data)')
    parser.add_argument('--clip_id', type=int, default=0,
                        help='Clip ID to visualize (default: 0)')

    args = parser.parse_args()

    visualize_sample(args.data_dir, args.clip_id)
