from typing import Tuple, Dict, Any, Optional, List

import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


def load_image(
    data_path: str,
    flag: int,
) -> Tuple[torch.Tensor, np.ndarray]:
    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Image file not found: {data_path}")
    if flag not in (0, 1):
        raise ValueError(f"flag must be 0 or 1, got {flag}")

    image = sitk.ReadImage(data_path)
    image = sitk.GetArrayFromImage(image)

    if image.size == 0:
        raise ValueError(f"Loaded image is empty: {data_path}")

    if flag == 1:
        image = np.clip(image, -250, 400)

    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean) / (std + 1e-8)
    original = image
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return tensor, original


class MMRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        mode: str,
        transform: Optional[torch.nn.Module] = None,
        fusion: str = 'late',
    ) -> None:
        if not os.path.isfile(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if mode not in ('CT', 'CP', 'CPC'):
            raise ValueError(f"mode must be 'CT', 'CP', or 'CPC', got '{mode}'")

        self.data = pd.read_excel(csv_path)
        if len(self.data) == 0:
            raise ValueError(f"CSV file is empty: {csv_path}")

        required_columns = ['id', 'CT_Path', 'PET_Path', 'MMR_label']
        missing = [col for col in required_columns if col not in self.data.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        self.transform = transform
        self.mode = mode
        self.fusion = fusion
        self.clinic_drop_columns = [
            'id', 'CT_Path', 'PET_Path',
            'MMR(dMMR：错配修复功能缺陷，1种及以上阴性；pMMR：功能完整，均阳性)',
            'MMR_label',
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        index = index % len(self.data)
        sample = self.data.iloc[index]
        label = sample['MMR_label']
        patient_id = sample['id']

        if self.mode == 'CT':
            CT_data, _ = load_image(sample['CT_Path'], flag=1)
            if self.transform:
                CT_data = self.transform(CT_data)
            return CT_data, label, patient_id

        elif self.mode == 'CP':
            CT_data, _ = load_image(sample['CT_Path'], flag=1)
            PET_data, _ = load_image(sample['PET_Path'], flag=0)
            image = torch.cat((CT_data, PET_data), dim=0)
            if self.transform:
                image = self.transform(image)
            CT_out = image[0].unsqueeze(0)
            PET_out = image[1].unsqueeze(0)
            return CT_out, PET_out, label, patient_id

        elif self.mode == 'CPC':
            clinic = sample.drop(self.clinic_drop_columns, errors='ignore')
            clinic_tensor = torch.tensor(clinic.astype(float).values).float()
            CT_data, _ = load_image(sample['CT_Path'], flag=1)
            PET_data, _ = load_image(sample['PET_Path'], flag=0)
            image = torch.cat((CT_data, PET_data), dim=0)
            if self.transform:
                image = self.transform(image)
            CT_out = image[0].unsqueeze(0)
            PET_out = image[1].unsqueeze(0)
            return CT_out, PET_out, label, patient_id, clinic_tensor


if __name__ == '__main__':
    dataset = MMRDataset(
        csv_path='./Dataset/train/train3_MMR.xlsx',
        mode='CPC',
    )
    print(f'Dataset size: {len(dataset)}')
    sample = dataset[0]
    print(f'Sample types: {[type(s) for s in sample]}')
