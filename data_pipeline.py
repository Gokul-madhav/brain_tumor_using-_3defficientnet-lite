
"""
01_data_pipeline.py
Brain Tumor Segmentation (BraTS2021)
PyTorch + MONAI

This is a production-ready starter pipeline.
"""

import os
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from monai.transforms import (
    Compose, EnsureChannelFirstd, NormalizeIntensityd,
    RandFlipd, RandRotate90d, RandGaussianNoised,
    RandScaleIntensityd, RandShiftIntensityd, RandGaussianSmoothd,
    CropForegroundd, Resized, ToTensord
)

TARGET_SHAPE = (128,128,128)
BATCH_SIZE = 2

def load_nifti(path):
    return nib.load(str(path)).get_fdata().astype(np.float32)

def remap_mask(mask):
    out = np.zeros_like(mask, dtype=np.uint8)
    out[mask==1]=1
    out[mask==2]=2
    out[mask==4]=3
    return out

class BraTSDataset(Dataset):
    def __init__(self, root, transform=None):
        self.root = Path(root)
        self.cases = sorted([p for p in self.root.iterdir() if p.is_dir()])
        self.transform = transform

    def __len__(self):
        return len(self.cases)

    def __getitem__(self, idx):
        c = self.cases[idx]
        pid = c.name

        t1 = load_nifti(c/f"{pid}_t1.nii.gz")
        t1ce = load_nifti(c/f"{pid}_t1ce.nii.gz")
        t2 = load_nifti(c/f"{pid}_t2.nii.gz")
        flair = load_nifti(c/f"{pid}_flair.nii.gz")
        seg = load_nifti(c/f"{pid}_seg.nii.gz")

        image = np.stack([t1,t1ce,t2,flair],axis=0)
        seg = remap_mask(seg)

        sample = {"image":image,"label":seg}

        if self.transform:
            sample = self.transform(sample)

        return sample

train_transform = Compose([
    EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image","label"], source_key="image"),
    Resized(
        keys="image",
        spatial_size=TARGET_SHAPE,
        mode="trilinear"
    ),

    Resized(
        keys="label",
        spatial_size=TARGET_SHAPE,
        mode="nearest"
    ),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image","label"], prob=0.5, spatial_axis=2),
    RandRotate90d(keys=["image","label"], prob=0.5),
    RandGaussianNoised(keys="image", prob=0.15, std=0.05),
    RandGaussianSmoothd(keys="image", prob=0.15),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.3),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.3),
    ToTensord(keys=["image","label"])
])

val_transform = Compose([
    EnsureChannelFirstd(keys="label", channel_dim="no_channel"),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image","label"], source_key="image"),
    Resized(keys="image", spatial_size=TARGET_SHAPE, mode="trilinear"),
    Resized(keys="label", spatial_size=TARGET_SHAPE, mode="nearest"),
    ToTensord(keys=["image","label"])
])

from torch.utils.data import random_split

def create_dataloaders(data_dir):

    dataset = BraTSDataset(
        data_dir,
        transform=None
    )

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_ds.dataset.transform = train_transform
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    return train_loader, val_loader

if __name__ == "__main__":
    DATASET = "dataset/BraTS2021_Training_Data"

    train_loader, _ = create_dataloaders(DATASET)

    batch = next(iter(train_loader))

    print("Images :", batch["image"].shape)
    print("Masks  :", batch["label"].shape)
    print("Unique Labels :", torch.unique(batch["label"]))
