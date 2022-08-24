import os
import os.path as osp
import argparse
import numpy as np
import SimpleITK as sitk
import shutil
from pathlib import Path
from settings import loader_settings


if __name__ == '__main__':
    input_path = loader_settings['InputPath']  # Path for the input
    output_path = '/nnunet_data'

    file_name_list = os.listdir(input_path)  # List of files in the input
    file_path_list = [os.path.join(input_path, f) for f in file_name_list]

    for fil in file_path_list:
        if '.nii.gz' in fil: # the suffix is .nii.gz
            out_name = os.path.basename(fil).replace('.nii.gz', '_0000.nii.gz')
            shutil.copyfile(fil, os.path.join(output_path, out_name))

        else: # the suffix is not .nii.gz
            file_name = os.path.basename(fil)
            base_file_name = file_name.split('.')[0]
            # suffix_name = file_name.replace(base_file_name, '')
            file_sitk_img = sitk.ReadImage(fil)
            sitk.WriteImage(file_sitk_img, os.path.join(output_path, base_file_name + '_0000.nii.gz'))

    print("Done")