import os
import json
import csv
from pathlib import Path
from typing import Dict, List

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def save_normalization_params(params: Dict[str, float], save_path: str):
    """Save height normalization parameters to JSON."""
    with open(save_path, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"Saved normalization params to {save_path}")


def load_normalization_params(load_path: str) -> Dict[str, float]:
    """Load height normalization parameters from JSON."""
    with open(load_path, 'r') as f:
        params = json.load(f)
    return params


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """Compute evaluation metrics.

    Args:
        pred: (B, 1, H, W) predictions
        target: (B, 1, H, W) targets

    Returns:
        Dict with metrics: l1, rmse, median_ae
    """
    with torch.no_grad():
        # L1 (MAE)
        l1 = torch.abs(pred - target).mean().item()

        # RMSE
        mse = torch.pow(pred - target, 2).mean().item()
        rmse = np.sqrt(mse)

        # Median absolute error
        abs_errors = torch.abs(pred - target).flatten()
        median_ae = torch.median(abs_errors).item()

    return {
        'l1': l1,
        'rmse': rmse,
        'median_ae': median_ae
    }


def visualize_prediction(
    images: Dict[str, torch.Tensor],
    target: torch.Tensor,
    prediction: torch.Tensor,
    save_path: str,
    denormalize_fn=None
):
    """Create 7-image strip visualization.

    Args:
        images: Dict with 5 input images (B, 3, 256, 256)
        target: Ground truth DSM (B, 1, 32, 32)
        prediction: Predicted DSM (B, 1, 32, 32)
        save_path: Path to save the visualization
        denormalize_fn: Function to denormalize heights
    """
    # Take first sample from batch
    ortho = images['ortho'][0].cpu().permute(1, 2, 0).numpy()
    north = images['north'][0].cpu().permute(1, 2, 0).numpy()
    east = images['east'][0].cpu().permute(1, 2, 0).numpy()
    south = images['south'][0].cpu().permute(1, 2, 0).numpy()
    west = images['west'][0].cpu().permute(1, 2, 0).numpy()

    target_img = target[0, 0].cpu().numpy()
    pred_img = prediction[0, 0].cpu().detach().numpy()

    # Denormalize if function provided
    if denormalize_fn is not None:
        target_img = denormalize_fn(torch.tensor(target_img)).numpy()
        pred_img = denormalize_fn(torch.tensor(pred_img)).numpy()

    # Create figure with 7 subplots
    fig, axes = plt.subplots(1, 7, figsize=(21, 3))

    # Plot RGB images
    axes[0].imshow(ortho)
    axes[0].set_title('Ortho')
    axes[0].axis('off')

    axes[1].imshow(north)
    axes[1].set_title('North')
    axes[1].axis('off')

    axes[2].imshow(east)
    axes[2].set_title('East')
    axes[2].axis('off')

    axes[3].imshow(south)
    axes[3].set_title('South')
    axes[3].axis('off')

    axes[4].imshow(west)
    axes[4].set_title('West')
    axes[4].axis('off')

    # Plot target and prediction with same colormap
    vmin = min(target_img.min(), pred_img.min())
    vmax = max(target_img.max(), pred_img.max())

    im1 = axes[5].imshow(target_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[5].set_title('Target DSM')
    axes[5].axis('off')
    plt.colorbar(im1, ax=axes[5], fraction=0.046)

    im2 = axes[6].imshow(pred_img, cmap='viridis', vmin=vmin, vmax=vmax)
    axes[6].set_title('Predicted DSM')
    axes[6].axis('off')
    plt.colorbar(im2, ax=axes[6], fraction=0.046)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


class CSVLogger:
    """Simple CSV logger for training metrics."""

    def __init__(self, log_path: str, fieldnames: List[str]):
        self.log_path = log_path
        self.fieldnames = fieldnames

        # Create file and write header
        with open(log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    def log(self, row: Dict[str, any]):
        """Append a row to the CSV."""
        with open(self.log_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)


class WarmupCosineScheduler:
    """Learning rate scheduler with warmup and cosine annealing."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        lr_min: float = 1e-6,
        lr_max: float = 1e-4
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.current_epoch = 0

    def step(self):
        """Update learning rate."""
        if self.current_epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.lr_min + (self.lr_max - self.lr_min) * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.lr_min + (self.lr_max - self.lr_min) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.current_epoch += 1
        return lr

    def get_last_lr(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']
