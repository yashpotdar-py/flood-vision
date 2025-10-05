import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from datasets.floodnet_dataset_4class import FloodNetDataset4Class
from models.deeplab_model import get_deeplab_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_COLORS = {
    0: (0, 0, 0),        # background
    1: (255, 0, 0),      # flooded building
    2: (255, 255, 0),    # flooded road
    3: (0, 0, 255)       # water
}


def mask_to_rgb(mask):
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls, color in CLASS_COLORS.items():
        rgb[mask == cls] = color
    return rgb


def visualize_predictions(model_path, sample_count=5):
    ROOT = r"C:\Users\yashy\projects\flood-vision\data\FloodNet-Supervised_v1.0"
    val_img = os.path.join(ROOT, "val/val-org-img")
    val_mask = os.path.join(ROOT, "val/val-label-img")

    dataset = FloodNetDataset4Class(val_img, val_mask, resize=(512, 512))
    model = get_deeplab_model(num_classes=4)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE).eval()

    for idx in range(sample_count):
        img, mask = dataset[idx]
        img_t = img.unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            pred = model(img_t)
            pred = torch.argmax(F.softmax(pred, dim=1),
                                dim=1).squeeze().cpu().numpy()

        img_np = img.permute(1, 2, 0).cpu().numpy()
        mask_np = mask.cpu().numpy()

        # Normalize for visualization
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Ground Truth")
        plt.imshow(mask_to_rgb(mask_np))
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Prediction")
        plt.imshow(mask_to_rgb(pred))
        plt.axis("off")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    visualize_predictions("checkpoint_deeplab_4class.pth", sample_count=30)
