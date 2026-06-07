from typing import Optional, Union

import os
import gc
import SimpleITK as sitk


def segment_colon(
    ct_path: str,
    output_path: str,
) -> None:
    if isinstance(ct_path, str):
        if not os.path.isfile(ct_path):
            raise FileNotFoundError(f"CT file not found: {ct_path}")

    from totalsegmentator.python_api import totalsegmentator
    import nibabel as nib

    if isinstance(ct_path, str):
        totalsegmentator(ct_path, output_path, roi_subset=["colon"])
    else:
        output_img = totalsegmentator(ct_path, roi_subset=["colon"])
        nib.save(output_img, output_path)

    if isinstance(output_path, str) and not os.path.exists(output_path):
        print(f"Warning: segmentation output not found at {output_path}")


def batch_segment_colon(
    ct_dir: str,
    save_dir: str,
) -> None:
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")

    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for ct_file in os.listdir(ct_dir):
        if not ct_file.endswith('.nii.gz'):
            continue

        count += 1
        input_path = os.path.join(ct_dir, ct_file)
        output_name = ct_file.replace('.nii.gz', '')
        output_path = os.path.join(save_dir, output_name)

        segment_colon(input_path, output_path)
        print(f'Colon segmentation done for case {count}: {ct_file}')
        gc.collect()

    if count == 0:
        print(f"Warning: no .nii.gz files found in {ct_dir}")


if __name__ == '__main__':
    batch_segment_colon(
        ct_dir='./data/CT',
        save_dir='./data/ColonSeg',
    )
