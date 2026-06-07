from typing import Optional, List, Dict

import os
import random

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import WeightedRandomSampler


def set_seed(seed: int = 42) -> None:
    if not isinstance(seed, int):
        raise TypeError(f"seed must be an integer, got {type(seed)}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_weighted_sampler(
    csv_path: str,
    label_column: str = 'MMR_label',
) -> WeightedRandomSampler:
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    labels = pd.read_excel(csv_path)[label_column]

    if len(labels) == 0:
        raise ValueError(f"No labels found in column '{label_column}'")

    class_counts = np.bincount(labels)
    if len(class_counts) < 2:
        raise ValueError(f"Need at least 2 classes, found {len(class_counts)}")

    zero_classes = [i for i, c in enumerate(class_counts) if c == 0]
    if zero_classes:
        raise ValueError(f"Class(es) {zero_classes} have zero samples")

    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]
    weights_tensor = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(
        weights_tensor,
        num_samples=len(weights_tensor),
        replacement=True,
    )


def get_default_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomRotation(180),
    ])


if __name__ == '__main__':
    set_seed(42)
    print(f'CUDA available: {torch.cuda.is_available()}')
    sampler = create_weighted_sampler('./Dataset/train/train3_MMR.xlsx')
    print(f'Sampler created with {sampler.num_samples} samples')
