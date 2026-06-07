import os
import gc
import time
import multiprocessing
from typing import Tuple, Optional, List

import SimpleITK as sitk


def resample_spacing(
    sitk_image_or_path,
    new_spacing: Tuple[float, float, float] = (1, 1, 1),
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    if isinstance(new_spacing, (list, tuple)):
        if len(new_spacing) != 3:
            raise ValueError(f"new_spacing must have 3 elements, got {len(new_spacing)}")
        if any(s <= 0 for s in new_spacing):
            raise ValueError(f"new_spacing values must be positive, got {new_spacing}")

    if isinstance(sitk_image_or_path, str):
        if not os.path.isfile(sitk_image_or_path):
            raise FileNotFoundError(f"Image file not found: {sitk_image_or_path}")
        sitk_image = sitk.ReadImage(sitk_image_or_path)
    else:
        sitk_image = sitk_image_or_path

    euler3d = sitk.Euler3DTransform()
    x_size, y_size, z_size = sitk_image.GetSize()
    x_spacing, y_spacing, z_spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    direction = sitk_image.GetDirection()

    new_size_x = round(x_size * x_spacing / new_spacing[0])
    new_size_y = round(y_size * y_spacing / new_spacing[1])
    new_size_z = round(z_size * z_spacing / new_spacing[2])
    new_size = (new_size_x, new_size_y, new_size_z)

    if any(s == 0 for s in new_size):
        raise ValueError(
            f"Computed new_size contains zero: {new_size}. "
            f"Original size=({x_size}, {y_size}, {z_size}), "
            f"Original spacing=({x_spacing}, {y_spacing}, {z_spacing}), "
            f"New spacing={new_spacing}"
        )

    resampled = sitk.Resample(
        sitk_image, new_size, euler3d, interpolator, origin, new_spacing, direction
    )
    return resampled


def batch_resample(
    images_dir: str,
    save_dir: str,
    new_spacing: Tuple[float, float, float] = (1, 1, 1),
    interpolator=sitk.sitkLinear,
) -> None:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    os.makedirs(save_dir, exist_ok=True)

    files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    if len(files) == 0:
        print(f"Warning: no .nii.gz files found in {images_dir}")
        return

    for f in files:
        start_time = time.perf_counter()
        input_path = os.path.join(images_dir, f)
        resampled = resample_spacing(
            input_path, new_spacing=new_spacing, interpolator=interpolator
        )
        output_path = os.path.join(save_dir, f)
        sitk.WriteImage(resampled, output_path)
        del resampled
        gc.collect()
        elapsed = time.perf_counter() - start_time
        print(f'Resampled {f} in {elapsed:.2f}s')


def batch_resample_parallel(
    images_dir: str,
    save_dir: str,
    new_spacing: Tuple[float, float, float] = (1, 1, 1),
    interpolator=sitk.sitkLinear,
    num_processes: Optional[int] = None,
) -> None:
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if num_processes is None:
        num_processes = multiprocessing.cpu_count()
    if num_processes <= 0:
        raise ValueError(f"num_processes must be positive, got {num_processes}")

    os.makedirs(save_dir, exist_ok=True)

    def _process_one(filename: str) -> None:
        start = time.perf_counter()
        input_path = os.path.join(images_dir, filename)
        resampled = resample_spacing(
            input_path, new_spacing=new_spacing, interpolator=interpolator
        )
        output_path = os.path.join(save_dir, filename)
        sitk.WriteImage(resampled, output_path)
        del resampled
        gc.collect()
        elapsed = time.perf_counter() - start
        print(f'Resampled {filename} in {elapsed:.2f}s')

    files = [f for f in os.listdir(images_dir) if f.endswith('.nii.gz')]
    if len(files) == 0:
        print(f"Warning: no .nii.gz files found in {images_dir}")
        return

    pool = multiprocessing.Pool(processes=num_processes)
    pool.map(_process_one, files)
    pool.close()
    pool.join()


if __name__ == '__main__':
    batch_resample(
        images_dir='./data/CT_cropped',
        save_dir='./data/CT_resampled',
        new_spacing=(1, 1, 1),
    )
