from typing import Optional, Tuple

import os
from data_processing.dicom_to_nii import dicom_to_nii, convert_dicom_to_nii, format_patient_id
from data_processing.pet_suv import PET_SUV, convert_pet_to_suv
from data_processing.segmentation import segment_colon, batch_segment_colon
from data_processing.registration import rigid_registration, batch_registration
from data_processing.roi_extraction import (
    extract_pet_colon_roi, extract_pet_roi,
    remove_small_connected_domains, expand_roi,
)
from data_processing.resample import resample_spacing, batch_resample, batch_resample_parallel
from data_processing.crop import crop_around_roi, crop_image_with_roi_center
from data_processing.build_dataset import build_dataset_csv, split_dataset


def run_segmentation_pipeline(
    ct_dir: str,
    pet_dir: str,
    dicom_dir: Optional[str],
    output_dir: str,
    suv_threshold: float = 6.0,
    crop_size: Tuple[int, int, int] = (120, 120, 120),
    expansion_voxels: int = 10,
) -> None:
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")
    if not os.path.isdir(pet_dir):
        raise FileNotFoundError(f"PET directory not found: {pet_dir}")
    if dicom_dir and not os.path.isdir(dicom_dir):
        raise FileNotFoundError(f"DICOM directory not found: {dicom_dir}")

    os.makedirs(output_dir, exist_ok=True)

    pet_suv_dir = os.path.join(output_dir, 'PET_SUV')
    if dicom_dir:
        print("[Step 1] Converting PET to SUV...")
        convert_pet_to_suv(pet_dir, dicom_dir, pet_suv_dir)
    else:
        pet_suv_dir = pet_dir

    print("[Step 2] Segmenting colon on CT...")
    colon_seg_dir = os.path.join(output_dir, 'ColonSeg')
    batch_segment_colon(ct_dir, colon_seg_dir)

    print("[Pipeline] Core functions ready.")


if __name__ == '__main__':
    run_segmentation_pipeline(
        ct_dir='./data/CT',
        pet_dir='./data/PET',
        dicom_dir='./data/DICOM',
        output_dir='./data/output',
    )
