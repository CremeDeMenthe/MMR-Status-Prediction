from typing import Optional

import os
import numpy as np
import SimpleITK as sitk


class PET_SUV:
    def __init__(self, df, nii: sitk.Image) -> None:
        if df is None:
            raise ValueError("DICOM dataframe cannot be None")
        if nii is None:
            raise ValueError("PET NIfTI image cannot be None")
        self.df = df
        self.nii = nii

    @staticmethod
    def dicom_hhmmss(t) -> float:
        t = str(t)
        if len(t) == 5:
            t = '0' + t
        if len(t) < 6:
            raise ValueError(f"Invalid DICOM time format: {t}")

        h_t = float(t[0:2])
        m_t = float(t[2:4])
        s_t = float(t[4:-1])
        total_seconds = h_t * 3600 + m_t * 60 + s_t
        return total_seconds

    def PET_SUV(self, df, nii: sitk.Image) -> np.ndarray:
        pet = sitk.GetArrayFromImage(nii)
        if pet.size == 0:
            raise ValueError("PET image array is empty")

        PW = df.get((0x0010, 0x1030)).value
        decay = df.get((0x0054, 0x1102)).value
        start_time = df.get((0x0008, 0x0032)).value

        radiopharma_seq = df.RadiopharmaceuticalInformationSequence[0]
        radio_start_time = str(radiopharma_seq['RadiopharmaceuticalStartTime'].value)
        dose = radiopharma_seq['RadionuclideTotalDose'].value
        half_life = str(radiopharma_seq['RadionuclideHalfLife'].value)

        start_time = self.dicom_hhmmss(start_time)
        radio_start_time = self.dicom_hhmmss(radio_start_time)

        if float(half_life) == 0:
            raise ValueError("RadionuclideHalfLife cannot be zero")

        time_diff = float(start_time - radio_start_time)
        decay_factor = pow(2, -time_diff / float(half_life))
        decay_dose = float(dose) * decay_factor

        if decay_dose == 0:
            raise ValueError("Decayed dose is zero, cannot compute SUV factor")

        factor = (PW * 1000) / decay_dose
        pet_SUV = pet * factor
        return pet_SUV


def convert_pet_to_suv(
    pet_nii_dir: str,
    dicom_data_dir: str,
    save_dir: str,
) -> None:
    if not os.path.isdir(pet_nii_dir):
        raise FileNotFoundError(f"PET NIfTI directory not found: {pet_nii_dir}")
    if not os.path.isdir(dicom_data_dir):
        raise FileNotFoundError(f"DICOM data directory not found: {dicom_data_dir}")

    import pydicom
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for pet_file in os.listdir(pet_nii_dir):
        if not pet_file.endswith('.nii.gz'):
            continue

        nii_name = os.path.basename(pet_file).replace('.nii.gz', '')
        for folder_name in os.listdir(dicom_data_dir):
            if nii_name == folder_name:
                count += 1
                patient_path = os.path.join(dicom_data_dir, folder_name)

                for j in os.listdir(patient_path):
                    sub_path = os.path.join(patient_path, j)
                    if not os.path.isdir(sub_path):
                        continue

                    for files in os.listdir(sub_path):
                        instance_path = os.path.join(sub_path, files)
                        if not os.path.isdir(instance_path):
                            continue

                        dicoms = os.listdir(instance_path)
                        if len(dicoms) == 0:
                            continue

                        dicom = pydicom.dcmread(os.path.join(instance_path, dicoms[0]))
                        if dicom.Modality == 'PT':
                            PET = sitk.ReadImage(os.path.join(pet_nii_dir, pet_file))
                            suv = PET_SUV(dicom, PET)
                            pet_suv = suv.PET_SUV(dicom, PET)

                            pet_suv_img = sitk.GetImageFromArray(pet_suv)
                            pet_suv_img.SetSpacing(PET.GetSpacing())
                            pet_suv_img.SetOrigin(PET.GetOrigin())
                            pet_suv_img.SetDirection(PET.GetDirection())

                            save_path = os.path.join(save_dir, pet_file)
                            sitk.WriteImage(pet_suv_img, save_path)
                            print(f'SUV conversion done for patient {count}: {pet_file}')


if __name__ == '__main__':
    convert_pet_to_suv(
        pet_nii_dir='./data/PET',
        dicom_data_dir='./data/DICOM',
        save_dir='./data/PET_SUV',
    )
