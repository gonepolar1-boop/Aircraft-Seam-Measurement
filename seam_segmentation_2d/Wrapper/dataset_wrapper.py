import os
import cv2
import torch
import numpy as np

from torch.utils.data import Dataset, random_split
from seam_segmentation_2d.Common.util import (
    BASELINE_EXPERIMENT_CONFIG,
    IMAGE_DIR_PATH,
    MASK_DIR_PATH,
)

class SeamDataset(Dataset):
    def __init__(
        self,
        image_dir_path=IMAGE_DIR_PATH,
        mask_directory_path=MASK_DIR_PATH,
        img_size=BASELINE_EXPERIMENT_CONFIG["img_size"]
    ):
        self.image_dir_path = image_dir_path
        self.mask_directory_path = mask_directory_path
        self.img_size = img_size
        self.image_names = sorted(os.listdir(image_dir_path))
        self.mask_names = sorted(os.listdir(mask_directory_path))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img = cv2.imread(os.path.join(self.image_dir_path, self.image_names[idx]), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(self.mask_directory_path, self.mask_names[idx]), cv2.IMREAD_GRAYSCALE)
        if self.img_size is not None:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)
        return torch.tensor(np.expand_dims(img, axis=0), dtype=torch.float32), torch.tensor(np.expand_dims(mask, axis=0), dtype=torch.float32)

    def split_dataset(
        self, 
        train_ratio=BASELINE_EXPERIMENT_CONFIG["train_ratio"], 
        val_ratio=BASELINE_EXPERIMENT_CONFIG["val_ratio"],
        seed=BASELINE_EXPERIMENT_CONFIG["seed"]
    ):
        dataset_size = len(self)
        if dataset_size == 0:
            raise ValueError("Dataset is empty, cannot split.")
        if train_ratio < 0 or val_ratio < 0:
            raise ValueError("train_ratio and val_ratio must be non-negative.")
        if train_ratio + val_ratio > 1:
            raise ValueError("train_ratio + val_ratio must be less than or equal to 1.")

        train_size, val_size = int(train_ratio * dataset_size), int(val_ratio * dataset_size)
        if train_ratio > 0 and train_size == 0:
            train_size = 1
        if val_ratio > 0 and val_size == 0 and dataset_size - train_size > 0:
            val_size = 1
        if train_size + val_size > dataset_size:
            val_size = max(dataset_size - train_size, 0)

        unused_size = dataset_size - train_size - val_size
        generator = torch.Generator().manual_seed(seed)
        train_dataset, val_dataset, _ = random_split(
            self,
            [train_size, val_size, unused_size],
            generator=generator
        )
        return train_dataset, val_dataset
