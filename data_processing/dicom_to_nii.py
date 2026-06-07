from typing import Tuple, List, Optional

import os
import SimpleITK as sitk


def format_patient_id(patient_id: str) -> str:
    if not isinstance(patient_id, str) or len(patient_id.strip()) == 0:
        raise ValueError("patient_id must be a non-empty string")

    formatted = patient_id.strip()
    formatted = formatted.upper()
    formatted = formatted.replace('-', '_')
    formatted = formatted.replace(' ', '_')
    return formatted


def dicom_to_nii(
    dicom_folder: str,
) -> Tuple[sitk.Image, Tuple[int, ...], Tuple[float, ...]]:
    if not os.path.isdir(dicom_folder):
        raise FileNotFoundError(f"DICOM folder not found: {dicom_folder}")

    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_folder)

    if len(dicom_names) == 0:
        raise ValueError(f"No DICOM files found in: {dicom_folder}")

    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    if image.GetSize() == (0, 0, 0):
        raise ValueError(f"Loaded DICOM image has zero size: {dicom_folder}")

    nii_image = sitk.Cast(image, sitk.sitkFloat32)
    size = nii_image.GetSize()
    voxel_size = nii_image.GetSpacing()
    return nii_image, size, voxel_size


def convert_dicom_to_nii(
    dicom_data_dir: str,
    save_dir: str,
) -> None:
    if not os.path.isdir(dicom_data_dir):
        raise FileNotFoundError(f"DICOM data directory not found: {dicom_data_dir}")

    os.makedirs(save_dir, exist_ok=True)
    num = 0
    for patient in os.listdir(dicom_data_dir):
        num += 1
        patient_path = os.path.join(dicom_data_dir, patient)
        if not os.path.isdir(patient_path):
            continue

        for series in os.listdir(patient_path):
            series_path = os.path.join(patient_path, series)
            if not os.path.isdir(series_path):
                continue

            for instance in os.listdir(series_path):
                instance_path = os.path.join(series_path, instance)
                if not os.path.isdir(instance_path):
                    continue

                dicoms = os.listdir(instance_path)
                if len(dicoms) == 0:
                    print(f"Warning: empty instance folder: {instance_path}")
                    continue

                import pydicom
                dcm_path = os.path.join(instance_path, dicoms[0])
                img = pydicom.dcmread(dcm_path)
                series_description = getattr(img, 'SeriesDescription', '')
                modality = getattr(img, 'Modality', '')

                if modality == 'CT' and 'Lung' in series_description:
                    save_path = os.path.join(save_dir, 'CT_Lung', patient + '.nii.gz')
                    img_nii, _, _ = dicom_to_nii(instance_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sitk.WriteImage(img_nii, save_path)
                    print(num, patient, save_path)

                elif modality == 'CT':
                    save_path = os.path.join(save_dir, 'CT', patient + '.nii.gz')
                    img_nii, _, _ = dicom_to_nii(instance_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sitk.WriteImage(img_nii, save_path)
                    print(num, patient, save_path)

                elif modality == 'PT':
                    save_path = os.path.join(save_dir, 'PET', patient + '.nii.gz')
                    img_nii, _, _ = dicom_to_nii(instance_path)
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    sitk.WriteImage(img_nii, save_path)
                    print(num, patient, save_path)


if __name__ == '__main__':
    convert_dicom_to_nii(
        dicom_data_dir='./data/DICOM',
        save_dir='./data/NIfTI',
    )
