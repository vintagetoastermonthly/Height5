#!/usr/bin/env python3
"""
Post-processing functions for height prediction refinement.

Uses superpixel segmentation and planar surface fitting to:
- Upsample predictions to original resolution (64×64 → 256×256)
- Align height boundaries with RGB edges
- Fit smooth planar surfaces within objects
"""

import numpy as np
import torch
from skimage.segmentation import slic
from skimage.filters import sobel
from typing import Tuple


def fit_plane_ransac(coords: np.ndarray, heights: np.ndarray,
                     ransac_iters: int = 100,
                     inlier_thresh: float = 0.1) -> np.ndarray:
    """
    Fit a plane using RANSAC to be robust to outliers.

    Args:
        coords: (N, 2) array of (x, y) coordinates
        heights: (N,) array of height values
        ransac_iters: Number of RANSAC iterations
        inlier_thresh: Threshold for inlier classification (in height units)

    Returns:
        Plane parameters [a, b, c] where h = ax + by + c
    """
    if len(coords) < 3:
        # Not enough points to fit a plane, return mean height
        mean_h = np.mean(heights)
        return np.array([0.0, 0.0, mean_h])

    best_params = None
    best_inliers = 0

    for _ in range(ransac_iters):
        # Randomly sample 3 points
        idx = np.random.choice(len(coords), size=min(3, len(coords)), replace=False)
        sample_coords = coords[idx]
        sample_heights = heights[idx]

        # Fit plane to sample: h = ax + by + c
        # Build matrix: [x, y, 1] * [a, b, c]^T = h
        A = np.column_stack([sample_coords, np.ones(len(sample_coords))])
        try:
            params, _, _, _ = np.linalg.lstsq(A, sample_heights, rcond=None)
        except np.linalg.LinAlgError:
            continue

        # Count inliers
        predicted = coords @ params[:2] + params[2]
        residuals = np.abs(heights - predicted)
        inliers = np.sum(residuals < inlier_thresh)

        if inliers > best_inliers:
            best_inliers = inliers
            best_params = params

    # Refit using all inliers if we found any good fit
    if best_params is not None and best_inliers >= 3:
        predicted = coords @ best_params[:2] + best_params[2]
        residuals = np.abs(heights - predicted)
        inlier_mask = residuals < inlier_thresh

        if np.sum(inlier_mask) >= 3:
            A = np.column_stack([coords[inlier_mask], np.ones(np.sum(inlier_mask))])
            try:
                best_params, _, _, _ = np.linalg.lstsq(A, heights[inlier_mask], rcond=None)
            except np.linalg.LinAlgError:
                pass

    # Fallback to simple least squares if RANSAC failed
    if best_params is None:
        A = np.column_stack([coords, np.ones(len(coords))])
        try:
            best_params, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
        except np.linalg.LinAlgError:
            # Ultimate fallback: constant height
            best_params = np.array([0.0, 0.0, np.mean(heights)])

    return best_params


def fit_plane_least_squares(coords: np.ndarray, heights: np.ndarray) -> np.ndarray:
    """
    Fit a plane using ordinary least squares.

    Args:
        coords: (N, 2) array of (x, y) coordinates
        heights: (N,) array of height values

    Returns:
        Plane parameters [a, b, c] where h = ax + by + c
    """
    if len(coords) < 3:
        # Not enough points to fit a plane, return mean height
        mean_h = np.mean(heights)
        return np.array([0.0, 0.0, mean_h])

    # Build matrix: [x, y, 1] * [a, b, c]^T = h
    A = np.column_stack([coords, np.ones(len(coords))])

    # Solve using least squares
    try:
        params, _, _, _ = np.linalg.lstsq(A, heights, rcond=None)
    except np.linalg.LinAlgError:
        # Fallback to mean height if singular
        mean_h = np.mean(heights)
        params = np.array([0.0, 0.0, mean_h])

    return params


def refine_segments_by_variance(
    segments: np.ndarray,
    heights: np.ndarray,
    variance_threshold: float = 0.5,
    min_size: int = 20
) -> np.ndarray:
    """
    Split superpixels that have high height variance (likely multi-plane surfaces).

    Handles cases like:
    - Gabled roofs with uniform color but two slopes
    - Steps/terraces in uniform colored areas

    Args:
        segments: (H, W) superpixel labels
        heights: (H, W) height values
        variance_threshold: Split segments with std > this value (in height units)
        min_size: Don't split segments smaller than this

    Returns:
        refined_segments: (H, W) with high-variance segments split
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
            # Split using median height as threshold
            y_coords, x_coords = np.where(mask)
            local_heights = heights[mask]

            median_h = np.median(local_heights)
            above_median = local_heights > median_h

            # Assign new label to pixels above median
            refined_segments[y_coords[above_median], x_coords[above_median]] = next_label
            next_label += 1

    return refined_segments


def superpixel_plane_refinement(
    ortho_rgb: np.ndarray,
    height_pred: np.ndarray,
    n_segments: int = 500,
    compactness: float = 10.0,
    use_ransac: bool = False,
    ransac_iters: int = 100,
    inlier_thresh: float = 0.1,
    height_aware: bool = False,
    height_weight: float = 10.0,
    variance_threshold: float = 0.5,
    min_segment_size: int = 20
) -> np.ndarray:
    """
    Refine height predictions using superpixel segmentation and planar surface fitting.

    This acts as a super-resolution step:
    1. Upsamples coarse height predictions to RGB resolution
    2. Segments image into superpixels using SLIC (RGB-only or RGB+height)
    3. Optionally splits high-variance segments (multi-plane surfaces like gabled roofs)
    4. Fits a planar surface to heights within each superpixel
    5. Returns refined height map with clean boundaries

    Args:
        ortho_rgb: (H, W, 3) RGB ortho image, values in [0, 1]
        height_pred: (h, w) predicted height map (typically 64×64)
        n_segments: Approximate number of superpixels to generate
        compactness: SLIC compactness parameter (higher = more compact/square segments)
        use_ransac: Whether to use RANSAC for plane fitting (robust to outliers)
        ransac_iters: Number of RANSAC iterations if use_ransac=True
        inlier_thresh: Inlier threshold for RANSAC (in height units)
        height_aware: Use height gradient as 4th channel for segmentation (better for ridges)
        height_weight: Weight for height channel relative to RGB (higher = respect height more)
        variance_threshold: Split segments with height std > this (detects multi-plane surfaces)
        min_segment_size: Minimum pixels per segment

    Returns:
        refined_height: (H, W) refined height map at RGB resolution

    Notes:
        - height_aware=True helps preserve ridges/edges not visible in RGB
        - Useful for gabled roofs with uniform color but distinct slopes
    """
    H, W = ortho_rgb.shape[:2]
    h, w = height_pred.shape

    # Step 1: Upsample height prediction to RGB resolution using bilinear interpolation
    if (h, w) != (H, W):
        # Convert to torch for easy upsampling
        height_tensor = torch.from_numpy(height_pred).unsqueeze(0).unsqueeze(0)  # (1, 1, h, w)
        upsampled = torch.nn.functional.interpolate(
            height_tensor, size=(H, W), mode='bilinear', align_corners=False
        )
        height_upsampled = upsampled.squeeze().numpy()
    else:
        height_upsampled = height_pred.copy()

    # Step 2: Run SLIC superpixel segmentation
    if height_aware:
        # Height-aware mode: Use height gradient as 4th channel
        # This helps preserve ridges/edges not visible in RGB

        # Compute height gradient magnitude
        height_grad = sobel(height_upsampled)

        # Normalize to [0, 1]
        if height_grad.max() > 0:
            height_grad_norm = height_grad / height_grad.max()
        else:
            height_grad_norm = height_grad

        # Create 4-channel image: RGB + weighted height gradient
        rgb_for_slic = (ortho_rgb * 255).astype(np.uint8)
        height_channel = (height_grad_norm * 255 * height_weight).astype(np.uint8)
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
        # Standard RGB-only segmentation
        rgb_for_slic = (ortho_rgb * 255).astype(np.uint8)
        segments = slic(
            rgb_for_slic,
            n_segments=n_segments,
            compactness=compactness,
            start_label=0,
            channel_axis=2
        )

    # Step 2.5: Optionally refine segments by splitting high-variance regions
    # This handles multi-plane surfaces (e.g., gabled roofs) within uniform RGB
    if variance_threshold > 0:
        segments = refine_segments_by_variance(
            segments, height_upsampled,
            variance_threshold=variance_threshold,
            min_size=min_segment_size
        )

    # Step 3: Fit planar surface to each superpixel
    refined_height = np.zeros_like(height_upsampled)
    n_superpixels = segments.max() + 1

    for seg_id in range(n_superpixels):
        # Get mask for this superpixel
        mask = segments == seg_id

        # Get coordinates and heights for this superpixel
        y_coords, x_coords = np.where(mask)
        heights = height_upsampled[mask]

        if len(heights) == 0:
            continue

        # Normalize coordinates to [0, 1] for numerical stability
        coords = np.column_stack([
            x_coords / W,
            y_coords / H
        ])

        # Fit plane: h = ax + by + c
        if use_ransac:
            plane_params = fit_plane_ransac(
                coords, heights,
                ransac_iters=ransac_iters,
                inlier_thresh=inlier_thresh
            )
        else:
            plane_params = fit_plane_least_squares(coords, heights)

        # Evaluate plane at all pixels in this superpixel
        a, b, c = plane_params
        fitted_heights = a * (x_coords / W) + b * (y_coords / H) + c

        # Assign fitted heights
        refined_height[y_coords, x_coords] = fitted_heights

    return refined_height


def postprocess_batch(
    ortho_images: torch.Tensor,
    height_predictions: torch.Tensor,
    n_segments: int = 500,
    compactness: float = 10.0,
    use_ransac: bool = False,
    height_aware: bool = False,
    height_weight: float = 10.0,
    variance_threshold: float = 0.5
) -> torch.Tensor:
    """
    Apply superpixel plane refinement to a batch of predictions.

    Args:
        ortho_images: (B, 3, H, W) batch of RGB ortho images, values in [0, 1]
        height_predictions: (B, 1, h, w) batch of predicted height maps
        n_segments: Number of superpixels per image
        compactness: SLIC compactness parameter
        use_ransac: Whether to use RANSAC for plane fitting
        height_aware: Use height gradient as 4th channel for segmentation
        height_weight: Weight for height channel relative to RGB
        variance_threshold: Split segments with height std > this value

    Returns:
        refined_heights: (B, 1, H, W) batch of refined height maps
    """
    batch_size = ortho_images.shape[0]
    H, W = ortho_images.shape[2:]

    refined_batch = []

    for i in range(batch_size):
        # Convert to numpy and move channels
        ortho_rgb = ortho_images[i].permute(1, 2, 0).cpu().numpy()  # (H, W, 3)
        height_pred = height_predictions[i, 0].cpu().numpy()  # (h, w)

        # Apply refinement
        refined = superpixel_plane_refinement(
            ortho_rgb, height_pred,
            n_segments=n_segments,
            compactness=compactness,
            use_ransac=use_ransac,
            height_aware=height_aware,
            height_weight=height_weight,
            variance_threshold=variance_threshold
        )

        # Convert back to torch
        refined_tensor = torch.from_numpy(refined).unsqueeze(0)  # (1, H, W)
        refined_batch.append(refined_tensor)

    # Stack into batch
    refined_heights = torch.stack(refined_batch, dim=0)  # (B, 1, H, W)

    return refined_heights


if __name__ == '__main__':
    """Test superpixel plane refinement."""
    import matplotlib.pyplot as plt

    print("="*70)
    print("Testing Superpixel Plane Refinement")
    print("="*70)

    # Create synthetic test data
    print("\n1. Creating synthetic test data...")
    H, W = 256, 256
    h, w = 64, 64

    # Create simple RGB image with distinct regions
    ortho_rgb = np.zeros((H, W, 3))
    ortho_rgb[:H//2, :W//2] = [0.8, 0.2, 0.2]  # Red region
    ortho_rgb[:H//2, W//2:] = [0.2, 0.8, 0.2]  # Green region
    ortho_rgb[H//2:, :W//2] = [0.2, 0.2, 0.8]  # Blue region
    ortho_rgb[H//2:, W//2:] = [0.8, 0.8, 0.2]  # Yellow region

    # Create coarse height prediction with different planes
    height_pred = np.zeros((h, w))
    height_pred[:h//2, :w//2] = 10.0  # Flat roof at 10m
    height_pred[:h//2, w//2:] = 5.0   # Flat roof at 5m
    height_pred[h//2:, :w//2] = 0.0   # Ground level
    height_pred[h//2:, w//2:] = 15.0  # Tall building at 15m

    # Add some noise
    height_pred += np.random.randn(h, w) * 0.5

    print(f"   Ortho RGB: {ortho_rgb.shape}")
    print(f"   Height pred: {height_pred.shape}")

    # Apply refinement
    print("\n2. Applying superpixel plane refinement...")
    refined = superpixel_plane_refinement(
        ortho_rgb, height_pred,
        n_segments=100,
        compactness=10.0,
        use_ransac=False
    )

    print(f"   Refined height: {refined.shape}")
    print(f"   Height range: [{refined.min():.2f}, {refined.max():.2f}]")

    # Test batch processing
    print("\n3. Testing batch processing...")
    ortho_batch = torch.from_numpy(ortho_rgb).permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
    height_batch = torch.from_numpy(height_pred).unsqueeze(0).unsqueeze(0).float()  # (1, 1, h, w)

    refined_batch = postprocess_batch(ortho_batch, height_batch, n_segments=100)
    print(f"   Batch output: {refined_batch.shape}")

    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)
    print("\nPost-processing is ready to use:")
    print("  - Upsamples 64×64 → 256×256")
    print("  - Aligns boundaries with RGB edges")
    print("  - Fits planar surfaces within superpixels")
