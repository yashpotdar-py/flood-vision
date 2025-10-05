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
    criterion = ComboLoss(num_classes=4, alpha=0.7)
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
