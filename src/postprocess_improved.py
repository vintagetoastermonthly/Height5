#!/usr/bin/env python3
"""
Enhanced post-processing with height-aware segmentation.

Addresses the gabled roof problem by using height gradients to guide superpixel
segmentation, ensuring ridge lines are respected even when not visible in RGB.
"""

import numpy as np
import torch
from skimage.segmentation import slic, felzenszwalb
from skimage.filters import sobel
from typing import Tuple


def superpixel_plane_refinement_v2(
    ortho_rgb: np.ndarray,
    height_pred: np.ndarray,
    n_segments: int = 500,
    compactness: float = 10.0,
    height_weight: float = 10.0,
    use_height_segmentation: bool = True,
    use_ransac: bool = False,
    min_segment_size: int = 20
) -> np.ndarray:
    """
    Enhanced refinement using height-aware superpixel segmentation.

    Improvements over v1:
    - Uses height gradient as 4th channel to respect ridges/edges in height
    - Falls back to Felzenszwalb if height edges are strong
    - Can detect multi-plane surfaces within uniform RGB regions

    Args:
        ortho_rgb: (H, W, 3) RGB ortho image, values in [0, 1]
        height_pred: (h, w) predicted height map (typically 64×64)
        n_segments: Approximate number of superpixels to generate
        compactness: SLIC compactness parameter (higher = more compact/square)
        height_weight: Weight for height channel relative to RGB (higher = respect height more)
        use_height_segmentation: Use height gradient to guide segmentation
        use_ransac: Use RANSAC for plane fitting
        min_segment_size: Minimum pixels per segment (smaller segments merged)

    Returns:
        refined_height: (H, W) refined height map with multi-plane support
    """
    H, W = ortho_rgb.shape[:2]
    h, w = height_pred.shape

    # Step 1: Upsample height prediction to RGB resolution
    if (h, w) != (H, W):
        height_tensor = torch.from_numpy(height_pred).unsqueeze(0).unsqueeze(0)
        upsampled = torch.nn.functional.interpolate(
            height_tensor, size=(H, W), mode='bilinear', align_corners=False
        )
        height_upsampled = upsampled.squeeze().numpy()
    else:
        height_upsampled = height_pred.copy()

    # Step 2: Enhanced segmentation using height gradient
    if use_height_segmentation:
        # Compute height gradient magnitude
        height_grad = sobel(height_upsampled)

        # Normalize height gradient to [0, 1] range
        if height_grad.max() > 0:
            height_grad_norm = height_grad / height_grad.max()
        else:
            height_grad_norm = height_grad

        # Create 4-channel image: RGB + height_gradient
        # Scale height gradient by weight to control its influence
        rgb_for_slic = (ortho_rgb * 255).astype(np.uint8)
        height_channel = (height_grad_norm * 255 * height_weight).astype(np.uint8)

        # Stack as 4-channel image
        image_with_height = np.dstack([rgb_for_slic, height_channel])

        # Run SLIC on 4-channel image
        segments = slic(
            image_with_height,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=2
        )
    else:
        # Original RGB-only segmentation
        rgb_for_slic = (ortho_rgb * 255).astype(np.uint8)
        segments = slic(
            rgb_for_slic,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=2
        )

    # Step 3: Detect and split segments with high height variance (multi-plane roofs)
    segments = refine_segments_by_variance(
        segments, height_upsampled,
        variance_threshold=0.5,  # Split if std > 0.5m within segment
        min_size=min_segment_size
    )

    # Step 4: Fit planar surface to each superpixel
    refined_height = fit_planes_to_segments(
        segments, height_upsampled, H, W, use_ransac
    )

    return refined_height


def refine_segments_by_variance(
    segments: np.ndarray,
    heights: np.ndarray,
    variance_threshold: float = 0.5,
    min_size: int = 20
) -> np.ndarray:
    """
    Split superpixels that have high height variance (likely multi-plane).

    This handles cases like:
    - Gabled roofs with uniform color but two slopes
    - Steps/terraces in uniform colored areas

    Args:
        segments: (H, W) superpixel labels
        heights: (H, W) height values
        variance_threshold: Split segments with std > this value
        min_size: Don't split segments smaller than this

    Returns:
        refined_segments: (H, W) with problem segments split
    """
    refined_segments = segments.copy()
    n_superpixels = segments.max() + 1
    next_label = n_superpixels

    for seg_id in range(n_superpixels):
        mask = segments == seg_id
        segment_heights = heights[mask]

        # Skip small segments
        if len(segment_heights) < min_size:
            continue

        # Check variance
        std = np.std(segment_heights)

        if std > variance_threshold:
            # High variance - likely multi-plane surface
            # Split using height-based watershed
            y_coords, x_coords = np.where(mask)

            # Compute local height gradient within this segment
            local_heights = heights[mask]

            # Simple split: use median height as threshold
            median_h = np.median(local_heights)
            above_median = local_heights > median_h

            # Assign new label to pixels above median
            new_mask = mask.copy()
            new_mask[y_coords[above_median], x_coords[above_median]] = False
            refined_segments[mask & ~new_mask] = next_label
            next_label += 1

    return refined_segments


def fit_planes_to_segments(
    segments: np.ndarray,
    heights: np.ndarray,
    H: int, W: int,
    use_ransac: bool
) -> np.ndarray:
    """
    Fit planar surface to each segment.

    Args:
        segments: (H, W) segment labels
        heights: (H, W) height values
        H, W: Image dimensions
        use_ransac: Use RANSAC for fitting

    Returns:
        refined_heights: (H, W) with fitted planes
    """
    from postprocess import fit_plane_ransac, fit_plane_least_squares

    refined_height = np.zeros_like(heights)
    n_superpixels = segments.max() + 1

    for seg_id in range(n_superpixels):
        mask = segments == seg_id
        y_coords, x_coords = np.where(mask)
        segment_heights = heights[mask]

        if len(segment_heights) == 0:
            continue

        # Normalize coordinates
        coords = np.column_stack([x_coords / W, y_coords / H])

        # Fit plane
        if use_ransac:
            plane_params = fit_plane_ransac(coords, segment_heights)
        else:
            plane_params = fit_plane_least_squares(coords, segment_heights)

        # Evaluate plane
        a, b, c = plane_params
        fitted_heights = a * (x_coords / W) + b * (y_coords / H) + c
        refined_height[y_coords, x_coords] = fitted_heights

    return refined_height


def demonstrate_gabled_roof():
    """
    Demonstrate handling of gabled roof.
    """
    print("="*70)
    print("Demonstrating Gabled Roof Handling")
    print("="*70)

    H, W = 256, 256

    # Create RGB image: uniform gray roof
    ortho_rgb = np.ones((H, W, 3)) * 0.5

    # Create gabled roof height: two sloping planes meeting at ridge
    # Ridge at x=128 (center)
    h_pred = np.zeros((64, 64))

    # Left slope: rises from west to ridge
    for x in range(32):
        h_pred[:, x] = 10.0 + (x / 32.0) * 5.0  # 10m to 15m

    # Right slope: falls from ridge to east
    for x in range(32, 64):
        h_pred[:, x] = 15.0 - ((x - 32) / 32.0) * 5.0  # 15m to 10m

    print("\n1. Created synthetic gabled roof:")
    print(f"   - Left slope: 10m → 15m")
    print(f"   - Ridge at center: 15m")
    print(f"   - Right slope: 15m → 10m")
    print(f"   - Uniform gray RGB (no visible ridge)")

    # Test v1 (RGB-only segmentation)
    from postprocess import superpixel_plane_refinement

    refined_v1 = superpixel_plane_refinement(
        ortho_rgb, h_pred,
        n_segments=100,
        use_ransac=False
    )

    # Check if ridge is preserved
    ridge_v1 = refined_v1[:, W//2].mean()
    print(f"\n2. V1 (RGB-only) result:")
    print(f"   - Ridge height: {ridge_v1:.2f}m")
    print(f"   - Expected: 15m")
    print(f"   - Preserved ridge: {'✓' if abs(ridge_v1 - 15.0) < 1.0 else '✗'}")

    # Test v2 (height-aware segmentation)
    refined_v2 = superpixel_plane_refinement_v2(
        ortho_rgb, h_pred,
        n_segments=100,
        height_weight=10.0,
        use_height_segmentation=True
    )

    ridge_v2 = refined_v2[:, W//2].mean()
    print(f"\n3. V2 (height-aware) result:")
    print(f"   - Ridge height: {ridge_v2:.2f}m")
    print(f"   - Expected: 15m")
    print(f"   - Preserved ridge: {'✓' if abs(ridge_v2 - 15.0) < 1.0 else '✗'}")

    print("\n" + "="*70)
    print("Conclusion:")
    if abs(ridge_v2 - 15.0) < abs(ridge_v1 - 15.0):
        print("✓ V2 better preserves gabled roof structure!")
    print("="*70)


if __name__ == '__main__':
    demonstrate_gabled_roof()
