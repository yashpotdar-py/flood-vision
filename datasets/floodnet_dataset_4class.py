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
