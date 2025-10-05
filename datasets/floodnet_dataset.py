"""
datasets/floodnet_dataset.py

Custom PyTorch Dataset for FloodNet semantic segmentation.

This module provides a PyTorch Dataset implementation for loading and preprocessing
FloodNet dataset images and masks. It supports resizing, data augmentation, and
proper normalization for semantic segmentation tasks.

Usage:
    pip install torch torchvision pillow numpy
    
    from datasets.floodnet_dataset import FloodNetDataset
    
    # Create dataset instance
    dataset = FloodNetDataset(
        img_dir="data/FloodNet-Supervised_v1.0/train/train-org-img",
        mask_dir="data/FloodNet-Supervised_v1.0/train/train-label-img",
        resize=(512, 512),
        augment=True
    )
    
    # Get a sample
    img, mask = dataset[0]
    
    # Use with DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)
"""
import os
import torch
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF


class FloodNetDataset(Dataset):
    """
    PyTorch Dataset for FloodNet semantic segmentation.

    This dataset loader handles FloodNet image-mask pairs, applies resizing,
    optional data augmentation (horizontal/vertical flips and rotations), and
    normalizes images using ImageNet statistics.

    Args:
        img_dir (str): Directory containing RGB input images.
        mask_dir (str): Directory containing corresponding segmentation masks.
        resize (tuple): Desired output size as (height, width). Default: (512, 512).
        augment (bool): Whether to apply random data augmentation. Default: False.

    Raises:
        AssertionError: If the number of images and masks don't match.

    Returns:
        tuple: (image, mask) where:
            - image: Normalized RGB tensor of shape (3, H, W)
            - mask: Long tensor of shape (H, W) with class indices
    """

    def __init__(self, img_dir, mask_dir, resize=(512, 512), augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

        # Validate that image and mask counts match (important)
        assert len(self.img_files) == len(self.mask_files), \
            f"Image-Mask count mismatch: {len(self.img_files)} vs {len(self.mask_files)}"

        self.resize = resize
        self.augment = augment

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Total number of image-mask pairs.
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Load and preprocess a single image-mask pair.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) where:
                - image: Normalized RGB tensor of shape (3, H, W)
                - mask: Long tensor of shape (H, W) with class indices
        """
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Resize both image and mask for torch tensor conversion
        img = TF.resize(img, self.resize, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=Image.NEAREST)

        # Data augmentation for training
        if self.augment:
            if random.random() > 0.5:
                # Horizontal flip
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                # Vertical flip
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                # Random rotation (90, 180, or 270 degrees)
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # Convert to tensor and normalize image
        img = TF.to_tensor(img)
        # Standard ImageNet normalization (according to paper)
        # TODO: verify if the values for mean/std are correct for FloodNet
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask
