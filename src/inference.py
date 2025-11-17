import argparse
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset import HeightEstimationDataset
from model import HeightEstimationModel
from utils import load_normalization_params, visualize_prediction
from postprocess import postprocess_batch


@torch.no_grad()
def run_inference(args):
    """Run inference on validation set or all data."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    # Load normalization params
    if 'height_norm_params' in checkpoint:
        height_norm_params = checkpoint['height_norm_params']
    else:
        # Try to load from separate file
        norm_params_path = Path(args.checkpoint).parent / 'height_norm_params.json'
        if norm_params_path.exists():
            height_norm_params = load_normalization_params(str(norm_params_path))
        else:
            raise ValueError("Height normalization parameters not found")

    print(f"Height normalization: min={height_norm_params['min']:.2f}, "
          f"max={height_norm_params['max']:.2f}")

    # Create dataset
    dataset = HeightEstimationDataset(
        data_dir=args.data_dir,
        split=args.split,
        height_norm_params=height_norm_params
    )

    # Create model
    model = HeightEstimationModel(pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Run inference
    print(f"Running inference on {len(dataset)} samples...")

    for idx in tqdm(range(len(dataset))):
        images, target = dataset[idx]

        # Add batch dimension
        images_batch = {k: v.unsqueeze(0).to(device) for k, v in images.items()}
        target_batch = target.unsqueeze(0)

        # Predict
        prediction = model(images_batch)

        # Apply post-processing if requested
        if args.postprocess:
            # Extract ortho image for post-processing
            ortho_batch = images_batch['ortho']  # (1, 3, 256, 256)

            # Apply superpixel plane refinement
            prediction = postprocess_batch(
                ortho_batch,
                prediction,
                n_segments=args.n_segments,
                compactness=args.compactness,
                use_ransac=args.use_ransac,
                height_aware=args.height_aware,
                height_weight=args.height_weight,
                variance_threshold=args.variance_threshold
            ).to(device)

        # Create denormalization function
        h_min = height_norm_params['min']
        h_max = height_norm_params['max']
        denorm_fn = lambda x: x * (h_max - h_min) + h_min

        # Denormalize prediction
        pred_denorm = denorm_fn(prediction[0, 0].cpu()).numpy()

        # Save prediction as image
        clip_info = dataset.clips[idx]
        clip_id = clip_info['clip_id']

        # Save as PNG (convert to 16-bit if possible)
        pred_uint16 = np.clip(pred_denorm, 0, 65535).astype(np.uint16)
        pred_img = Image.fromarray(pred_uint16, mode='I;16')
        pred_path = output_dir / f'{clip_id}_predicted_dsm.png'
        pred_img.save(pred_path)

        # Optionally save visualization
        if args.save_visualizations:
            vis_path = output_dir / f'{clip_id}_visualization.png'
            visualize_prediction(
                {k: v.unsqueeze(0) for k, v in images.items()},
                target_batch,
                prediction,
                str(vis_path),
                denormalize_fn=denorm_fn
            )

    print(f"Inference complete! Results saved to {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference with trained model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='val',
                        choices=['train', 'val'],
                        help='Which split to run inference on')
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Directory to save predictions')
    parser.add_argument('--save_visualizations', action='store_true',
                        help='Save 7-image strip visualizations')

    # Post-processing arguments
    parser.add_argument('--postprocess', action='store_true',
                        help='Apply superpixel plane refinement for super-resolution')
    parser.add_argument('--n_segments', type=int, default=500,
                        help='Number of superpixels for SLIC segmentation')
    parser.add_argument('--compactness', type=float, default=10.0,
                        help='SLIC compactness parameter (higher = more compact segments)')
    parser.add_argument('--use_ransac', action='store_true',
                        help='Use RANSAC for plane fitting (robust to outliers)')
    parser.add_argument('--height_aware', action='store_true',
                        help='Use height gradient as 4th channel for segmentation (better for ridges/gabled roofs)')
    parser.add_argument('--height_weight', type=float, default=10.0,
                        help='Weight for height channel relative to RGB (only with --height_aware)')
    parser.add_argument('--variance_threshold', type=float, default=0.5,
                        help='Split segments with height std > this value (detects multi-plane surfaces)')

    args = parser.parse_args()

    run_inference(args)
