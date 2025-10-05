import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import os

from models.unet_baseline import get_unet_model
from datasets.floodnet_dataset import FloodNetDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_iou(preds, labels, num_classes):
    """Computes mean IoU across all classes."""
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)

    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls
        intersection = (pred_inds & label_inds).sum().item()
        union = (pred_inds | label_inds).sum().item()
        if union == 0:
            ious.append(float("nan"))
        else:
            ious.append(intersection / union)
    return np.nanmean(ious), ious


def evaluate(model, loader, num_classes=10):
    model.eval()
    total_correct, total_pixels = 0, 0
    total_iou, iou_list = 0, np.zeros(num_classes)

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            preds = torch.argmax(F.softmax(outputs, dim=1), dim=1)

            correct = (preds == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel()

            miou, class_iou = compute_iou(
                preds.cpu(), masks.cpu(), num_classes)
            iou_list += np.nan_to_num(class_iou)
            total_iou += miou

    pixel_acc = total_correct / total_pixels
    mean_iou = total_iou / len(loader)
    classwise_iou = iou_list / len(loader)
    return pixel_acc, mean_iou, classwise_iou


def main():
    # === Paths ===
    ROOT = "data\\FloodNet-Supervised_v1.0"
    val_img = os.path.join(ROOT, "val/val-org-img")
    val_mask = os.path.join(ROOT, "val/val-label-img")

    # === Dataset & Loader ===
    val_dataset = FloodNetDataset(
        val_img, val_mask, resize=(512, 512), augment=False)
    val_loader = DataLoader(val_dataset, batch_size=2,
                            shuffle=False, num_workers=2)

    # === Load model checkpoint ===
    model = get_unet_model(num_classes=10).to(DEVICE)
    model.load_state_dict(torch.load(
        "checkpoint_unet.pth", map_location=DEVICE))

    pixel_acc, mean_iou, classwise_iou = evaluate(
        model, val_loader, num_classes=10)

    print("\n===== Evaluation Results =====")
    print(f"Pixel Accuracy : {pixel_acc:.4f}")
    print(f"Mean IoU        : {mean_iou:.4f}")
    print("Class-wise IoU :")
    for i, iou in enumerate(classwise_iou):
        print(f"  Class {i:02d}: {iou:.4f}")


if __name__ == "__main__":
    main()
