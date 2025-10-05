"""
train/train_segmentation_4class.py

Trains a DeepLab segmentation model on the FloodNet dataset using 4 classes:
Background, Flooded Building, Flooded Road, Water.

Features:
    - Training with data augmentation
    - Validation monitoring
    - Learning rate scheduling (ReduceLROnPlateau)
    - Best model checkpoint saving based on validation loss
    - Combined Dice + Cross-Entropy loss (ComboLoss)

Usage:
    pip install torch torchvision numpy tqdm
    python -m train.train_segmentation_4class
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.floodnet_dataset_4class import FloodNetDataset4Class
from models.deeplab_model import get_deeplab_model
from utils.losses import ComboLoss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): Segmentation model.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion):
    """
    Validate the model on the validation set.

    Args:
        model (torch.nn.Module): Segmentation model.
        loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average validation loss.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Val", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def main():
    """
    Main training function. Loads datasets, initializes model and optimizer,
    and runs the training loop for 25 epochs with validation.
    """
    ROOT = "data\\FloodNet-Supervised_v1.0"
    train_img = os.path.join(ROOT, "train/train-org-img")
    train_mask = os.path.join(ROOT, "train/train-label-img")
    val_img = os.path.join(ROOT, "val/val-org-img")
    val_mask = os.path.join(ROOT, "val/val-label-img")

    train_ds = FloodNetDataset4Class(train_img, train_mask, augment=True)
    val_ds = FloodNetDataset4Class(val_img, val_mask, augment=False)
    train_loader = DataLoader(train_ds, batch_size=4,
                              shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)

    model = get_deeplab_model(num_classes=4).to(DEVICE)
    # Weighted for minority classes: [Background, Flooded Building, Flooded Road, Water]
    class_weights = [0.5, 4.0, 3.0, 2.0]
    criterion = ComboLoss(num_classes=4, alpha=0.7,
                          class_weights=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=3, factor=0.5)

    best_val_loss = float("inf")

    for epoch in range(25):
        print(f"\nEpoch {epoch+1}/25")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoint_deeplab_4class.pth")
            print("Saved best model checkpoint")


if __name__ == "__main__":
    main()
