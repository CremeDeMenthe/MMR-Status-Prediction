from typing import Tuple

import numpy as np
import SimpleITK as sitk


def crop_around_roi(
    image_array: np.ndarray,
    roi_array: np.ndarray,
    target_size: Tuple[int, int, int] = (96, 96, 96),
) -> np.ndarray:
    if image_array.ndim != 3:
        raise ValueError(f"image_array must be 3D, got {image_array.ndim}D")
    if roi_array.ndim != 3:
        raise ValueError(f"roi_array must be 3D, got {roi_array.ndim}D")
    if image_array.shape != roi_array.shape:
        raise ValueError(
            f"Shape mismatch: image {image_array.shape} vs roi {roi_array.shape}"
        )
    if len(target_size) != 3:
        raise ValueError(f"target_size must have 3 elements, got {len(target_size)}")
    if any(s <= 0 for s in target_size):
        raise ValueError(f"target_size values must be positive, got {target_size}")

    roi_index = np.where(roi_array != 0)
    if len(roi_index[0]) == 0:
        print("Warning: ROI is empty, returning center crop")
        center_z = image_array.shape[0] // 2
        center_y = image_array.shape[1] // 2
        center_x = image_array.shape[2] // 2
    else:
        min_z = np.min(roi_index[0])
        max_z = np.max(roi_index[0])
        min_y = np.min(roi_index[1])
        max_y = np.max(roi_index[1])
        min_x = np.min(roi_index[2])
        max_x = np.max(roi_index[2])
        center_z = (min_z + max_z) // 2
        center_y = (min_y + max_y) // 2
        center_x = (min_x + max_x) // 2

    half_z = target_size[0] // 2
    half_y = target_size[1] // 2
    half_x = target_size[2] // 2
    start_z = max(center_z - half_z, 0)
    start_y = max(center_y - half_y, 0)
    start_x = max(center_x - half_x, 0)

    cropped = image_array[
        start_z: start_z + target_size[0],
        start_y: start_y + target_size[1],
        start_x: start_x + target_size[2],
    ]

    actual_size = cropped.shape
    if actual_size != target_size:
        print(f"Warning: cropped size {actual_size} differs from target {target_size}")

    return cropped


def crop_image_with_roi_center(
    ref_image: sitk.Image,
    image: sitk.Image,
    roi_mask: sitk.Image,
    crop_size: Tuple[int, int, int],
) -> sitk.Image:
    if not isinstance(crop_size, (list, tuple)) or len(crop_size) != 3:
        raise ValueError(f"crop_size must be a tuple of 3 integers, got {crop_size}")
    if any(s <= 0 for s in crop_size):
        raise ValueError(f"crop_size values must be positive, got {crop_size}")

    image_array = sitk.GetArrayFromImage(image)
    roi_array = sitk.GetArrayFromImage(roi_mask)

    if image_array.shape != roi_array.shape:
        raise ValueError(
            f"Shape mismatch: image {image_array.shape} vs roi {roi_array.shape}"
        )

    non_zero = np.where(roi_array != 0)
    if len(non_zero[0]) == 0:
        print("Warning: ROI mask is empty, using image center for cropping")
        center_z = image_array.shape[0] // 2
        center_y = image_array.shape[1] // 2
        center_x = image_array.shape[2] // 2
    else:
        center_z = int(np.mean(non_zero[0]))
        center_y = int(np.mean(non_zero[1]))
        center_x = int(np.mean(non_zero[2]))

    half_z = crop_size[0] // 2
    half_y = crop_size[1] // 2
    half_x = crop_size[2] // 2

    start_z = max(center_z - half_z, 0)
    end_z = min(center_z + half_z, image_array.shape[0])
    start_y = max(center_y - half_y, 0)
    end_y = min(center_y + half_y, image_array.shape[1])
    start_x = max(center_x - half_x, 0)
    end_x = min(center_x + half_x, image_array.shape[2])

    cropped = image_array[start_z:end_z, start_y:end_y, start_x:end_x]
    cropped_img = sitk.GetImageFromArray(cropped)
    cropped_img.SetSpacing(image.GetSpacing())
    cropped_img.SetDirection(image.GetDirection())
    return cropped_img


if __name__ == '__main__':
    import os
    roi_dir = './data/ROI'
    ct_dir = './data/CT_resampled'

    if os.path.isdir(roi_dir) and os.path.isdir(ct_dir):
        for f in os.listdir(ct_dir):
            if not f.endswith('.nii.gz'):
                continue
            img = sitk.ReadImage(os.path.join(ct_dir, f))
            arr = sitk.GetArrayFromImage(img)
            print(f'{f}: shape={arr.shape}, spacing={img.GetSpacing()}')
