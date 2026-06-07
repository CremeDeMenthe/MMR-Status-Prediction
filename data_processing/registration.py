import os
import gc
from typing import Tuple

import SimpleITK as sitk


def rigid_registration(
    ct_img: sitk.Image,
    pet_img: sitk.Image,
    ct_roi: sitk.Image,
) -> sitk.Image:
    if ct_img.GetDimension() != pet_img.GetDimension():
        raise ValueError(
            f"Dimension mismatch: CT {ct_img.GetDimension()}D vs PET {pet_img.GetDimension()}D"
        )
    if ct_img.GetDimension() != ct_roi.GetDimension():
        raise ValueError(
            f"Dimension mismatch: CT {ct_img.GetDimension()}D vs ROI {ct_roi.GetDimension()}D"
        )

    initial_transform = sitk.TranslationTransform(ct_img.GetDimension())

    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation()
    registration_method.SetMetricSamplingStrategy(sitk.ImageRegistrationMethod.REGULAR)
    registration_method.SetMetricSamplingPercentage(1.0)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=0.001,
        numberOfIterations=200,
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform)

    transform = registration_method.Execute(pet_img, ct_img)

    if registration_method.GetOptimizerStopConditionDescription() == '':
        print("Warning: optimizer did not converge")

    pet_roi = sitk.Resample(
        ct_roi, ct_img, transform,
        sitk.sitkNearestNeighbor, 0.0, ct_roi.GetPixelID(),
    )
    return pet_roi


def batch_registration(
    ct_dir: str,
    pet_dir: str,
    roi_dir: str,
    save_dir: str,
) -> None:
    if not os.path.isdir(ct_dir):
        raise FileNotFoundError(f"CT directory not found: {ct_dir}")
    if not os.path.isdir(pet_dir):
        raise FileNotFoundError(f"PET directory not found: {pet_dir}")
    if not os.path.isdir(roi_dir):
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")

    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for roi_file in os.listdir(roi_dir):
        ct_file = roi_file.replace('.uint16', '')
        pet_file = roi_file.replace('.uint16', '')

        ct_path = os.path.join(ct_dir, ct_file)
        pet_path = os.path.join(pet_dir, pet_file)
        roi_path = os.path.join(roi_dir, roi_file)

        if not os.path.isfile(ct_path):
            print(f"Warning: CT file not found: {ct_path}")
            continue
        if not os.path.isfile(pet_path):
            print(f"Warning: PET file not found: {pet_path}")
            continue

        if ct_file == pet_file and ct_file == roi_file.replace('.uint16', ''):
            count += 1
            ct_img = sitk.ReadImage(ct_path)
            pet_img = sitk.ReadImage(pet_path)
            ct_roi = sitk.ReadImage(roi_path)

            pet_roi = rigid_registration(ct_img, pet_img, ct_roi)
            output_path = os.path.join(save_dir, ct_file)
            sitk.WriteImage(pet_roi, output_path)
            print(f'Registration done for case {count}: {ct_file}')

            del ct_img, pet_img, pet_roi
            gc.collect()


if __name__ == '__main__':
    batch_registration(
        ct_dir='./data/CT',
        pet_dir='./data/PET',
        roi_dir='./data/ColonSeg',
        save_dir='./data/PET_ROI',
    )
