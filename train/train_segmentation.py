"""
train/train_segmentation.py

Trains a U-Net segmentation model on the FloodNet training set and validates
on the FloodNet validation set using 10 classes.

Computes:
    - Training loss per epoch
    - Validation loss and pixel accuracy per epoch
    - Saves best model checkpoint based on validation loss

Usage:
    pip install torch torchvision numpy tqdm
    python -m train.train_segmentation
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# import custom dataset and model
from datasets.floodnet_dataset import FloodNetDataset
from models.unet_baseline import get_unet_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, optimizer, criterion):
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): Segmentation model to train.
        loader (DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    # train loop for one epoch
    for imgs, masks in tqdm(loader, desc="Train", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


def validate_one_epoch(model, loader, criterion):
    """
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): Segmentation model to validate.
        loader (DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.

    Returns:
        (float, float): Tuple of average validation loss and pixel accuracy.
    """
    model.eval()
    total_loss = 0
    correct_pixels = 0
    total_pixels = 0
    # validation loop for one epoch
    with torch.no_grad():
        # disable gradient computation
        for imgs, masks in tqdm(loader, desc="Validate", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            # forward pass
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            total_loss += loss.item() * imgs.size(0)
            # compute accuracy
            preds = torch.argmax(outputs, dim=1)
            correct_pixels += (preds == masks).sum().item()
            total_pixels += masks.numel()
    # overall accuracy
    acc = correct_pixels / total_pixels
    return total_loss / len(loader.dataset), acc


def main():
    """
    Main training function. Loads the training and validation datasets,
    initializes the model, and runs the training loop for 10 epochs.
    Saves the best model checkpoint based on validation loss.
    """
    #  === Paths ===
    ROOT = "data\\FloodNet-Supervised_v1.0"
    train_img = os.path.join(ROOT, "train/train-org-img")
    train_mask = os.path.join(ROOT, "train/train-label-img")
    val_img = os.path.join(ROOT, "val/val-org-img")
    val_mask = os.path.join(ROOT, "val/val-label-img")

    # === Dataset & Loader ===
    train_dataset = FloodNetDataset(
        train_img, train_mask, resize=(512, 512), augment=True)
    val_dataset = FloodNetDataset(
        val_img, val_mask, resize=(512, 512), augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=4,
                            shuffle=False, num_workers=2)

    # === Model ===
    model = get_unet_model(num_classes=10).to(DEVICE)

    # === Criterion & Optimizer ===
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # === Training loop ===
    best_val_loss = float("inf")
    for epoch in range(10):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion)

        print(
            f"Epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "checkpoint_unet.pth")
            print("Saved new best model checkpoint")


if __name__ == "__main__":
    main()
