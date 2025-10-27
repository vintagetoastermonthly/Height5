#!/usr/bin/env python3
"""
Generate dummy image data for scale testing the Height5 dataloader.
Creates 200 complete image sets (6 files each) matching the real data structure.
"""

import os
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def create_dummy_rgb_image(size=(256, 256)):
    """Create a random RGB image."""
    # Generate random RGB values
    img_array = np.random.randint(0, 256, size=(size[0], size[1], 3), dtype=np.uint8)
    return Image.fromarray(img_array, 'RGB')


def create_dummy_dsm_image(size=(256, 256)):
    """Create a random 16-bit DSM image with realistic height distribution."""
    # Generate heights with some structure (gradient + noise)
    x = np.linspace(0, 100, size[1])
    y = np.linspace(0, 100, size[0])
    xx, yy = np.meshgrid(x, y)

    # Base gradient + random noise
    base_height = xx * 0.5 + yy * 0.3
    noise = np.random.randn(size[0], size[1]) * 10

    dsm = (base_height + noise).astype(np.float32)
    # Clip to reasonable range and convert to uint16
    dsm = np.clip(dsm, 0, 200)
    dsm_uint16 = (dsm * 256).astype(np.uint16)

    return Image.fromarray(dsm_uint16, mode='I;16')


def generate_dummy_dataset(output_dir, num_clips=200, start_id=1000):
    """
    Generate dummy dataset with proper naming convention.

    Args:
        output_dir: Directory to save dummy data
        num_clips: Number of complete clip sets to generate
        start_id: Starting clip ID (default 1000 to avoid conflicts with existing data)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_clips} dummy image sets in {output_dir}")
    print(f"Each set contains 6 files (1 DSM + 1 ortho + 4 obliques)")
    print(f"Total files to create: {num_clips * 6}")
    print()

    for i in tqdm(range(num_clips), desc="Creating dummy data"):
        clip_id = start_id + i
        clip_id_padded = f"{clip_id:04d}"

        # Create DSM (16-bit PNG)
        dsm_filename = f"{clip_id}_dsm.png"
        dsm_path = output_path / dsm_filename
        dsm_img = create_dummy_dsm_image()
        dsm_img.save(dsm_path)

        # Create ortho nadir image
        ortho_filename = f"{clip_id}_prop{clip_id_padded}_nadir.tiff.jpg"
        ortho_path = output_path / ortho_filename
        ortho_img = create_dummy_rgb_image()
        ortho_img.save(ortho_path, quality=95)

        # Create oblique images (north, south, east, west)
        for direction in ['north', 'south', 'east', 'west']:
            oblique_filename = f"{clip_id}_prop{clip_id_padded}_oblique-{direction}.tiff.jpg"
            oblique_path = output_path / oblique_filename
            oblique_img = create_dummy_rgb_image()
            oblique_img.save(oblique_path, quality=95)

    print(f"\nDone! Created {num_clips} complete clip sets")
    print(f"Clip IDs: {start_id} to {start_id + num_clips - 1}")

    # Verify by counting files
    dsm_count = len(list(output_path.glob("*_dsm.png")))
    jpg_count = len(list(output_path.glob("*.jpg")))
    print(f"\nVerification:")
    print(f"  DSM files: {dsm_count}")
    print(f"  JPG files: {jpg_count}")
    print(f"  Expected: {num_clips} DSM, {num_clips * 5} JPG")

    if dsm_count == num_clips and jpg_count == num_clips * 5:
        print("  ✓ All files created successfully!")
    else:
        print("  ✗ File count mismatch!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate dummy data for Height5 testing")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_dummy",
        help="Output directory for dummy data (default: data_dummy)"
    )
    parser.add_argument(
        "--num_clips",
        type=int,
        default=5000,
        help="Number of clip sets to generate (default: 200)"
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=1000,
        help="Starting clip ID to avoid conflicts (default: 1000)"
    )

    args = parser.parse_args()

    generate_dummy_dataset(
        output_dir=args.output_dir,
        num_clips=args.num_clips,
        start_id=args.start_id
    )
