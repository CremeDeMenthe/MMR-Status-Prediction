from typing import Optional

import os
import pandas as pd
from sklearn.model_selection import train_test_split


def build_dataset_csv(
    output_path: str,
    ct_dir: str,
    pet_dir: str,
) -> None:
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")
    if not os.path.isdir(pet_dir):
        raise FileNotFoundError(f"PET directory not found: {pet_dir}")

    ct_files = sorted([f for f in os.listdir(ct_dir) if f.endswith('.nii.gz')])
    pet_files = sorted([f for f in os.listdir(pet_dir) if f.endswith('.nii.gz')])

    if len(ct_files) == 0:
        raise ValueError(f"No .nii.gz files found in CT directory: {ct_dir}")
    if len(pet_files) == 0:
        raise ValueError(f"No .nii.gz files found in PET directory: {pet_dir}")
    if len(ct_files) != len(pet_files):
        print(f"Warning: CT has {len(ct_files)} files but PET has {len(pet_files)} files")

    df = pd.DataFrame(columns=['id', 'CT_Path', 'PET_Path'])
    df.to_excel(output_path, index=False)

    with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
        for ct_file, pet_file in zip(ct_files, pet_files):
            ct_full_path = os.path.join(ct_dir, ct_file)
            pet_full_path = os.path.join(pet_dir, pet_file)
            row = [ct_file, ct_full_path, pet_full_path]
            data = pd.DataFrame([row])
            data.to_excel(
                writer,
                header=False,
                index=False,
                sheet_name='Sheet1',
                startrow=writer.sheets['Sheet1'].max_row,
            )

    print(f"Dataset CSV saved to {output_path} with {len(ct_files)} entries")


def split_dataset(
    dataset_path: str,
    train_path: str,
    test_path: str,
    label_column: str,
) -> None:
    if not os.path.isfile(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

    data = pd.read_excel(dataset_path)

    if label_column not in data.columns:
        raise ValueError(
            f"Label column '{label_column}' not found. Available: {list(data.columns)}"
        )

    if len(data) < 2:
        raise ValueError(f"Dataset too small to split: {len(data)} samples")

    label_counts = data[label_column].value_counts()
    if len(label_counts) < 2:
        raise ValueError(
            f"Need at least 2 classes for stratified split, got {len(label_counts)}"
        )

    train_data, test_data = train_test_split(
        data,
        test_size=0.2,
        stratify=data[label_column],
        random_state=42,
    )
    train_data.to_excel(train_path, index=False)
    test_data.to_excel(test_path, index=False)
    print(f"Train label distribution:\n{train_data[label_column].value_counts()}")
    print(f"Test label distribution:\n{test_data[label_column].value_counts()}")


if __name__ == '__main__':
    dataset_csv = './data/dataset.xlsx'
    build_dataset_csv(
        output_path=dataset_csv,
        ct_dir='./data/CT_resampled',
        pet_dir='./data/PET_resampled',
    )
    split_dataset(
        dataset_path=dataset_csv,
        train_path='./data/train.xlsx',
        test_path='./data/test.xlsx',
        label_column='MMR_label',
    )
