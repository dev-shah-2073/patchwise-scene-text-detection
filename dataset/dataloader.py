import os
import time
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing import process_image_and_coords, image_to_patches_reshape
from ground_truth import modify_coords, generate_ground_truth


class PatchTextFullDataset:
    
    def __init__(self, root_dir, patch_size=48, train_ratio=0.8, seed=42):
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.train_ratio = train_ratio
        self.seed = seed

        self.images, self.targets = self._build_dataset()
        self._split_dataset()

    def _read_annotation(self, path):
        coordinates = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split(',')
                if parts[-1] == '##::ENGLISH' or len(parts) != 9:
                    continue

                coordinates.append(list(map(int, parts[:8])))

        return np.array(coordinates)

    def _build_dataset(self):
        image_dir = os.path.join(self.root_dir, "Image")
        ann_dir = os.path.join(self.root_dir, "Annotation")

        file_names = sorted(os.listdir(image_dir))

        images = []
        targets = []

        tic = time.time()
        for idx, filename in enumerate(file_names):
            image_path = os.path.join(image_dir, filename)
            ann_path = os.path.join(ann_dir, filename.replace('.jpg', '.txt'))

            coords = self._read_annotation(ann_path)

            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            transformed_image, transformed_coords = process_image_and_coords(img, coords)
            patches = image_to_patches_reshape(transformed_image, self.patch_size)
            boxes = modify_coords(transformed_coords)
            gt = generate_ground_truth(boxes)

            images.append(patches)
            targets.append(gt)

            if (idx + 1) % 10 == 0:
                print(f"{idx + 1} images processed")

        print(f"Dataset build time: {time.time() - tic:.2f}s")

        images = torch.stack(images)    # (N, 256, 3, 48, 48)
        targets = torch.stack(targets)  # (N, 256, 5)

        return images, targets

    def _split_dataset(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        num_samples = self.images.shape[0]
        num_train = int(self.train_ratio * num_samples)

        indices = torch.randperm(num_samples)
        train_idx = indices[:num_train]
        test_idx = indices[num_train:]

        self.train_images = self.images[train_idx]
        self.train_targets = self.targets[train_idx]
        self.test_images = self.images[test_idx]
        self.test_targets = self.targets[test_idx]


class PatchTensorDataset(Dataset):
    
    def __init__(self, images, targets):
        self.images = images
        self.targets = targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.targets[idx]


def build_dataloaders(root_dir, batch_size=16, num_workers=2, pin_memory=True):
    full = PatchTextFullDataset(root_dir)

    train_dataset = PatchTensorDataset(full.train_images, full.train_targets)
    test_dataset = PatchTensorDataset(full.test_images, full.test_targets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, test_loader
