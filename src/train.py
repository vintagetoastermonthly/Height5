import os
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_dataloaders
from model import HeightEstimationModel
from utils import (
    compute_metrics,
    visualize_prediction,
    save_normalization_params,
    CSVLogger,
    WarmupCosineScheduler,
    combined_loss
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    use_combined_loss: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for images, target in pbar:
        # Move to device
        images = {k: v.to(device) for k, v in images.items()}
        target = target.to(device)

        # Forward pass
        optimizer.zero_grad()
        prediction = model(images)

        # Compute loss
        if use_combined_loss:
            loss, loss_dict = combined_loss(prediction, target)
            pbar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'l1': f'{loss_dict["l1"]:.4f}',
                'grad': f'{loss_dict["gradient"]:.4f}'
            })
        else:
            loss = criterion(prediction, target)
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_metrics = {'l1': 0.0, 'rmse': 0.0, 'median_ae': 0.0}
    num_batches = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for images, target in pbar:
        # Move to device
        images = {k: v.to(device) for k, v in images.items()}
        target = target.to(device)

        # Forward pass
        prediction = model(images)
        loss = criterion(prediction, target)

        # Compute metrics
        metrics = compute_metrics(prediction, target)

        # Accumulate
        total_loss += loss.item()
        for k, v in metrics.items():
            all_metrics[k] += v
        num_batches += 1

        pbar.set_postfix({'loss': f'{loss.item():.6f}'})

    # Average metrics
    avg_loss = total_loss / num_batches
    avg_metrics = {k: v / num_batches for k, v in all_metrics.items()}
    avg_metrics['loss'] = avg_loss

    return avg_metrics


def train(args):
    """Main training function."""
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True)
    examples_dir = Path(args.examples_dir)
    examples_dir.mkdir(exist_ok=True)

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, height_norm_params = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=args.augment,
        drop_prob=args.drop_prob
    )

    # Save normalization parameters
    norm_params_path = checkpoint_dir / 'height_norm_params.json'
    save_normalization_params(height_norm_params, str(norm_params_path))

    # Create model
    print("Creating model...")
    model = HeightEstimationModel(pretrained=args.pretrained)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.epochs,
        lr_min=args.lr_min,
        lr_max=args.lr
    )

    # CSV logger
    csv_logger = CSVLogger(
        log_path=args.log_path,
        fieldnames=[
            'epoch', 'train_loss', 'val_loss', 'val_rmse',
            'val_median_ae', 'lr', 'time_epoch'
        ]
    )

    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0

    # Get validation samples for visualization (fixed set)
    val_iter = iter(val_loader)
    vis_images, vis_target = next(val_iter)

    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            use_combined_loss=True
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        val_loss = val_metrics['loss']

        # Update learning rate
        current_lr = scheduler.step()

        # Log metrics
        epoch_time = time.time() - epoch_start
        csv_logger.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_rmse': val_metrics['rmse'],
            'val_median_ae': val_metrics['median_ae'],
            'lr': current_lr,
            'time_epoch': epoch_time
        })

        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}, RMSE: {val_metrics['rmse']:.6f}, "
              f"Median AE: {val_metrics['median_ae']:.6f}")
        print(f"  LR: {current_lr:.6e}, Time: {epoch_time:.1f}s")

        # Save best model and generate visualizations
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            checkpoint_path = checkpoint_dir / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'height_norm_params': height_norm_params,
            }, checkpoint_path)
            print(f"  Saved best model (val_loss: {val_loss:.6f})")

            # Generate visualizations for 5 samples
            print("  Generating visualizations...")
            model.eval()
            with torch.no_grad():
                vis_images_device = {k: v.to(device) for k, v in vis_images.items()}
                vis_pred = model(vis_images_device)

                # Create denormalization function
                h_min = height_norm_params['min']
                h_max = height_norm_params['max']
                denorm_fn = lambda x: x * (h_max - h_min) + h_min

                # Save 5 examples
                for i in range(min(5, vis_images['ortho'].shape[0])):
                    sample_images = {k: v[i:i+1] for k, v in vis_images.items()}
                    sample_target = vis_target[i:i+1]
                    sample_pred = vis_pred[i:i+1]

                    save_path = examples_dir / f'example_{i}_epoch{epoch}.png'
                    visualize_prediction(
                        sample_images,
                        sample_target,
                        sample_pred,
                        str(save_path),
                        denormalize_fn=denorm_fn
                    )
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train height estimation model')

    # Data
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (reduce if OOM)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Enable rotation augmentation (90° increments)')
    parser.add_argument('--drop_prob', type=float, default=0.0,
                        help='Probability of dropping each input image (0.0-1.0, default: 0.0)')

    # Model
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained ResNet50')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Initial learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6,
                        help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of warmup epochs')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')

    # Output
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--examples_dir', type=str, default='examples',
                        help='Directory to save example visualizations')
    parser.add_argument('--log_path', type=str, default='training_log.csv',
                        help='Path to CSV log file')
    parser.add_argument('--save_every', type=int, default=50,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()

    train(args)
