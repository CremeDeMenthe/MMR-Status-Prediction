from typing import Tuple, Dict, Any

import SimpleITK as sitk
import numpy as np


def extract_pet_colon_roi(
    pet_img: sitk.Image,
    colon_mask: sitk.Image,
    suv_threshold: float = 4.0,
) -> Tuple[sitk.Image, sitk.Image]:
    if pet_img.GetSize() != colon_mask.GetSize():
        raise ValueError(
            f"Size mismatch: PET {pet_img.GetSize()} vs mask {colon_mask.GetSize()}"
        )
    if suv_threshold <= 0:
        raise ValueError(f"suv_threshold must be positive, got {suv_threshold}")

    roi = sitk.BinaryThreshold(
        colon_mask,
        lowerThreshold=0,
        upperThreshold=6,
        insideValue=0,
        outsideValue=1,
    )
    pet_arr = sitk.GetArrayFromImage(pet_img)
    roi_arr = sitk.GetArrayFromImage(roi)

    if pet_arr.shape != roi_arr.shape:
        raise ValueError(
            f"Array shape mismatch: PET {pet_arr.shape} vs ROI {roi_arr.shape}"
        )

    pet_roi = pet_arr * roi_arr

    pet_colon_img = sitk.GetImageFromArray(pet_roi)
    pet_colon_img.SetOrigin(pet_img.GetOrigin())
    pet_colon_img.SetSpacing(pet_img.GetSpacing())
    pet_colon_img.SetDirection(pet_img.GetDirection())

    tumor_mask = sitk.BinaryThreshold(
        pet_colon_img,
        lowerThreshold=0,
        upperThreshold=suv_threshold,
        insideValue=0,
        outsideValue=1,
    )
    return pet_colon_img, tumor_mask


def extract_pet_roi(
    pet_img: sitk.Image,
    roi_mask: sitk.Image,
) -> sitk.Image:
    pet_arr = sitk.GetArrayFromImage(pet_img)
    roi_arr = sitk.GetArrayFromImage(roi_mask)

    if pet_arr.shape != roi_arr.shape:
        raise ValueError(
            f"Shape mismatch: PET {pet_arr.shape} vs ROI {roi_arr.shape}"
        )

    if np.sum(roi_arr) == 0:
        print("Warning: ROI mask is empty, result will be all zeros")

    pet_roi = pet_arr * roi_arr

    result = sitk.GetImageFromArray(pet_roi)
    result.SetOrigin(pet_img.GetOrigin())
    result.SetSpacing(pet_img.GetSpacing())
    result.SetDirection(pet_img.GetDirection())
    return result


def remove_small_connected_domains(
    itk_mask: sitk.Image,
) -> Tuple[sitk.Image, Dict[int, Dict[str, Any]]]:
    mask_arr = sitk.GetArrayFromImage(itk_mask)
    if np.sum(mask_arr) == 0:
        print("Warning: input mask is empty, returning zeros")
        res_itk = sitk.GetImageFromArray(np.zeros_like(mask_arr))
        res_itk.SetOrigin(itk_mask.GetOrigin())
        res_itk.SetSpacing(itk_mask.GetSpacing())
        res_itk.SetDirection(itk_mask.GetDirection())
        return res_itk, {}

    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)
    num_labels = cc_filter.GetObjectCount()

    if num_labels == 0:
        print("Warning: no connected components found")
        res_itk = sitk.GetImageFromArray(np.zeros_like(mask_arr))
        res_itk.SetOrigin(itk_mask.GetOrigin())
        res_itk.SetSpacing(itk_mask.GetSpacing())
        res_itk.SetDirection(itk_mask.GetDirection())
        return res_itk, {}

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    area_dict = {}
    for label in range(1, num_labels + 1):
        area_dict[label] = lss_filter.GetNumberOfPixels(label)

    max_label = max(area_dict, key=area_dict.get)
    max_area = area_dict[max_label]

    np_output = sitk.GetArrayFromImage(output_mask)
    res_mask = np.zeros_like(np_output)
    res_mask[np_output == max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    info = {
        max_label: {
            'label': max_label,
            'area': max_area,
            'bounding_box': lss_filter.GetBoundingBox(max_label),
        }
    }
    return res_itk, info


def expand_roi(
    roi_image: sitk.Image,
    expansion_size: int = 10,
) -> sitk.Image:
    if expansion_size < 0:
        raise ValueError(f"expansion_size must be non-negative, got {expansion_size}")

    size = roi_image.GetSize()
    if size == (0, 0, 0):
        raise ValueError("Cannot expand an empty ROI image")

    dilated_roi = sitk.BinaryDilate(roi_image, [expansion_size] * 3)
    return dilated_roi


if __name__ == '__main__':
    import os
    ct_dir = './data/CT'
    pet_dir = './data/PET_SUV'
    roi_dir = './data/ColonSeg'

    if os.path.isdir(ct_dir) and os.path.isdir(roi_dir):
        for f in os.listdir(roi_dir):
            ct_img = sitk.ReadImage(os.path.join(ct_dir, f))
            roi_img = sitk.ReadImage(os.path.join(roi_dir, f))
            roi_cleaned, info = remove_small_connected_domains(roi_img)
            expanded = expand_roi(roi_cleaned, expansion_size=10)
            print(f'Processed {f}, ROI info: {info}')
