"""
datasets/floodnet_dataset_4class.py
FloodNet Dataset for 4-class semantic segmentation.
This dataset class loads FloodNet images and corresponding masks, remapping the original
class labels to 4 classes. Supports optional data augmentation and normalization.

"""
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

CLASS_MAP = {
    0: 0,  # background
    1: 1,  # building flooded
    2: 0,  # non-flooded building → background
    3: 2,  # road flooded
    4: 0,  # non-flooded road → background
    5: 3,  # water
    6: 0,  # tree → background
    7: 0,  # vehicle → background
    8: 0,  # pool → background
    9: 0,  # grass → background
}


class FloodNetDataset4Class(Dataset):
    """
    FloodNet Dataset for 4-class semantic segmentation.
    This dataset class loads FloodNet images and corresponding masks, remapping the original
    class labels to 4 classes. Supports optional data augmentation and normalization.
    The dataset expects:
        - img_dir: Directory containing RGB images
        - mask_dir: Directory containing corresponding mask images
    The class remaps original FloodNet labels to 4 classes using CLASS_MAP dictionary.
    Images and masks are matched by sorted order in their respective directories.
    Augmentations applied (when augment=True):
        - Resize to specified dimensions
        - Horizontal flip (p=0.5)
        - Vertical flip (p=0.3)
        - Random brightness/contrast adjustment (p=0.4)
        - Hue/saturation/value adjustment (p=0.3)
        - Gaussian blur (p=0.2)
        - Random 90° rotation (p=0.3)
        - ImageNet normalization
    Without augmentation, only resize and normalization are applied.
    Returns:
        Tuple of (image_tensor, mask_tensor) where:
            - image_tensor: [C, H, W] float32 tensor, normalized
            - mask_tensor: [H, W] long tensor with remapped class labels
    Usage:
        train_dataset = FloodNetDataset4Class(
            img_dir='data/train/images',
            mask_dir='data/train/masks',
            resize=(512, 512),
            augment=True
        )
        val_dataset = FloodNetDataset4Class(
            img_dir='data/val/images',
            mask_dir='data/val/masks',
            resize=(512, 512),
            augment=False
        )
    """

    def __init__(self, img_dir, mask_dir, resize=(512, 512), augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.imgs = sorted(os.listdir(img_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.augment = augment

        if self.augment:
            self.transform = A.Compose([
                A.Resize(*resize),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(p=0.2),
                A.RandomRotate90(p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(*resize),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.imgs)

    def remap_mask(self, mask):
        remapped = np.zeros_like(mask)
        for old, new in CLASS_MAP.items():
            remapped[mask == old] = new
        return remapped

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        mask = self.remap_mask(mask)

        augmented = self.transform(image=img, mask=mask)
        return augmented["image"], augmented["mask"].long()
