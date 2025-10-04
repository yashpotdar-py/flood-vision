"""
dataset/floodnet_dataset.py

Custom PyTorch Dataset for FloodNet semantic segmentation
Usage:
    pip install torch torchvision pillow numpy
    from datasets.floodnet_dataset import FloodNetDataset
    dataset = FloodNetDataset(img_dir="data/images", 
                            mask_dir="data/masks", 
                            resize=(512,512), 
                            augment=True)
    img, mask = dataset[0]  # get first sample
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

    Args:
        img_dir (str): Directory with RGB input images.
        mask_dir (str): Directory with corresponding segmentation masks.
        resize (tuple): Desired (height, width) for output tensors.
        augment (bool): Apply random flips/rotations if True.
    """

    def __init__(self, img_dir, mask_dir, resize=(512, 512), augment=False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_files = sorted(os.listdir(img_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        # validate that image and mask counts match (imp)
        assert len(self.img_files) == len(self.mask_files), \
            f"Image-Mask count mismatch: {len(self.img_files)} vs {len(self.mask_files)}"

        self.resize = resize
        self.augment = augment

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # resize both image and mask for torch tensor conversion
        img = TF.resize(img_path, self.resize, interpolation=Image.BILINEAR)
        mask = TF.resize(mask, self.resize, interpolation=Image.NEAREST)

        # data augmentation for training
        if self.augment:
            if random.random() > 0.5:
                # horizontal flip
                img = TF.hflip(img)
                mask = TF.hflip(mask)
            if random.random() > 0.5:
                # vertical flip
                img = TF.vflip(img)
                mask = TF.vflip(mask)
            if random.random() > 0.5:
                # random rotation
                angle = random.choice([90, 180, 270])
                img = TF.rotate(img, angle, interpolation=Image.BILINEAR)
                mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        # convert to tensor and normalize image
        img = TF.to_tensor(img)
        # standard ImageNet normalization (according to paper)
        # TODO: verify if the values for mean/std are correct for FloodNet
        img = TF.normalize(img,
                           mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])

        mask = torch.from_numpy(np.array(mask)).long()

        return img, mask
