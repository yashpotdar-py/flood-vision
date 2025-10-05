import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets.floodnet_dataset import FloodNetDataset
from models.unet_baseline import get_unet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_predictions(model_path, sample_count=5):
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
            pred = torch.argmax(F.softmax(pred, dim=1), dim=1).squeeze().cpu().numpy()

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
