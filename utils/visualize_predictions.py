"""
utils/visualize_predictions.py

Visualizes model predictions on the FloodNet validation dataset.
Loads a trained U-Net model and displays side-by-side comparisons of original images,

ground truth masks, and model predictions for semantic segmentation.
The script uses a 10-class segmentation scheme as defined in the FloodNet dataset.
Usage:
    python -m utils.visualize_predictions

    Or modify the model_path and sample_count in __main__ block before running.
Requirements:
    torch, numpy, matplotlib, datasets.floodnet_dataset, models.unet_baseline
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets.floodnet_dataset import FloodNetDataset
from models.unet_baseline import get_unet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def visualize_predictions(model_path, sample_count=5):
    """
    Visualize model predictions on FloodNet validation dataset.
    Loads a trained U-Net model and displays side-by-side comparisons of original images,
    ground truth masks, and model predictions for semantic segmentation.
    Args:
        model_path (str): Path to the trained model weights file (.pth)
        sample_count (int, optional): Number of validation samples to visualize. Defaults to 5.
    Usage:
        visualize_predictions("models/unet_best.pth", sample_count=10)
    Notes:
        - Expects FloodNet dataset structure at data/FloodNet-Supervised_v1.0
        - Resizes images to 512x512 for inference
        - Uses 'tab10' colormap for 10-class segmentation visualization
        - Model is automatically moved to available device (CPU/GPU)
    """
    ROOT = "data\\FloodNet-Supervised_v1.0"
    val_img = os.path.join(ROOT, "val/val-org-img")
    val_mask = os.path.join(ROOT, "val/val-label-img")

    dataset = FloodNetDataset(val_img, val_mask, resize=(512, 512))
    model = get_unet_model(num_classes=10)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    for idx in range(sample_count):
        img, mask = dataset[idx]
        img_t = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(img_t)
            pred = torch.argmax(F.softmax(pred, dim=1),
                                dim=1).squeeze().cpu().numpy()

        img_np = img.permute(1, 2, 0).numpy()
        mask_np = mask.numpy()

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original Image")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_np, cmap="tab10")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(pred, cmap="tab10")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualize_predictions("checkpoint_unet.pth", sample_count=3)
