# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)


# -------------------------------------------------- Imports --------------------------------------------------------

import os
from os.path import join
import shutil
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import numpy as np
import pandas as pd

from dipy.io.image import load_nifti, save_nifti

from collections import OrderedDict




# --------------------------------------------- ConvertLabels class --------------------------------------------------


class ConvertLabels:

    def __init__(self):
        #####################################################
        # SET THESE:
        self.target = 'rthal'
        self.target_code = 49
        self.project_root = '/Users/sa936/projects/capsnet'
        self.segs_csv = 'data/datasets_nnUNet/combined_outputs.csv'
        self.corrected_folder = 'data/images'
        #####################################################

        self.segs_list = make_image_list(join(self.project_root, self.segs_csv))

        for seg_path in tqdm(self.segs_list, desc='converting segmentation labels'):
            seg, affine = load_nifti(seg_path)
            seg = self.correct_labels(seg)

            folder, file = self.make_corrected_paths(seg_path, self.project_root, self.corrected_folder)

            os.makedirs(folder, exist_ok=True)
            save_nifti(join(folder, file), seg, affine)



    def correct_labels(self, seg):
        seg2 = np.zeros_like(seg)
        seg2[seg == self.target_code] = 1
        return seg2



    def make_corrected_paths(self, seg_path, project_root, corrected_folder):
        """
        Example path
        /Users/arman/projects/capsnet/data/images/137_S_0825/2007-10-29_12_12_41.0/aparc+aseg_brainbox.nii.gz
        """
        path_components = seg_path.split('/')
        subject, scan, file = path_components[-3], path_components[-2], path_components[-1]
        file = self.target + '.nii.gz'  # e.g. rhipp.nii.gz
        folder = join(project_root, corrected_folder, subject, scan)
        return folder, file



# --------------------------------------------- ConvertLabels class --------------------------------------------------

class MakeFolderTree():

    '''
    This is the folder tree we want to create:

        nnUNet_raw_data_base/nnUNet_raw_data/Task002_Heart/
    ├── dataset.json
    ├── imagesTr
    │   ├── rhipp_003_0000.nii.gz
    │   ├── rhipp_004_0000.nii.gz
    │   ├── ...
    ├── imagesTs
    │   ├── rhipp_001_0000.nii.gz
    │   ├── rhipp_002_0000.nii.gz
    │   ├── ...
    └── labelsTr
        ├── rhipp_003.nii.gz
        ├── rhipp_004.nii.gz
        ├── ...
    '''

    def __init__(self):

        project_root = '/Users/sa936/projects/capsnet'
        datasets_folder = 'data/datasets_nnUNet'
        train_inputs_csv = 'train_inputs.csv'
        train_outputs_csv = 'train_outputs.csv'
        valid_inputs_csv = 'valid_inputs.csv'
        valid_outputs_csv = 'valid_outputs.csv'

        task_name = 'rthal'
        nnUNet_raw_data = 'data/nnUNet_raw_data'

        imagesTr_list = make_image_list(join(project_root, datasets_folder, train_inputs_csv))
        labelsTr_list = make_image_list(join(project_root, datasets_folder, train_outputs_csv))
        imagesTs_list = make_image_list(join(project_root, datasets_folder, valid_inputs_csv))
        labelsTs_list = make_image_list(join(project_root, datasets_folder, valid_outputs_csv))

        imagesTr_folder = join(project_root, nnUNet_raw_data, task_name, 'imagesTr')
        labelsTr_folder = join(project_root, nnUNet_raw_data, task_name, 'labelsTr')
        imagesTs_folder = join(project_root, nnUNet_raw_data, task_name, 'imagesTs')
        labelsTs_folder = join(project_root, nnUNet_raw_data, task_name, 'labelsTs')


        os.makedirs(imagesTr_folder, exist_ok=True)
        os.makedirs(labelsTr_folder, exist_ok=True)
        os.makedirs(imagesTs_folder, exist_ok=True)
        os.makedirs(labelsTs_folder, exist_ok=True)

        equivalence_df = pd.DataFrame(columns=['ADNI', 'nnUNet'])

        i = 0
        _, equivalence_df = copy_images(task_name, imagesTr_list, imagesTr_folder, 'imagesTr', i, equivalence_df)
        i, equivalence_df = copy_labels(task_name, labelsTr_list, labelsTr_folder, 'labelsTr', i, equivalence_df)
        _, equivalence_df = copy_images(task_name, imagesTs_list, imagesTs_folder, 'imagesTs', i, equivalence_df)
        i, equivalence_df = copy_labels(task_name, labelsTs_list, labelsTs_folder, 'labelsTs', i, equivalence_df)

        equivalence_df.to_csv(join(project_root, datasets_folder, 'ADNI_nnUNet_table.csv'))








# ---------------------------------------------- Helper functions ---------------------------------------------------

def copy_images(task_name, images_list, images_folder, folder_name, i, equivalence_df):

    for path in tqdm(images_list, desc=f'copying {folder_name}'):
        new_path = join(images_folder, f'{task_name}_{i:04d}_0000.nii.gz')
        equivalence_row = pd.DataFrame(data={'ADNI': [path], 'nnUNet': [new_path]})
        equivalence_df = pd.concat([equivalence_df, equivalence_row])
        shutil.copy(path, new_path)
        i += 1

    return i, equivalence_df


def copy_labels(task_name, label_list, labels_folder, folder_name, i, equivalence_df):

    for path in tqdm(label_list, desc=f'copying {folder_name}'):
        new_path = join(labels_folder, f'{task_name}_{i:04d}.nii.gz')
        equivalence_row = pd.DataFrame(data={'ADNI': [path], 'nnUNet': [new_path]})
        equivalence_df = pd.concat([equivalence_df, equivalence_row])
        shutil.copy(path, new_path)
        i += 1

    return i, equivalence_df



def make_image_list(images_csv):
    images_df = pd.read_csv(images_csv, header=None)
    image_list = list(images_df.iloc[:, 0])
    return image_list



def make_files_list(root_folder):
    """
    Input:
    - root_folder: the root folder that contains sub-folders and files for which the list will be made

    Output:
    - list of paths to all files in the root folder and its sub-folders.
    """
    files_list = []
    for path, _, files in tqdm(os.walk(root_folder), desc='Making list of the corrected MRIs'):
        for file in files:
            files_list.append(join(root_folder, path, file))

    return files_list



def filter_files(file_list, structure):
    """
    Inputs:
        - files_list: list of file paths
        - structure: file paths will be filtered to contain this structure

    Output:
    - filtered file list: only paths that contain the structure are returned
    """
    return [file_name for file_name in file_list if structure in file_name]

# ------------------------------------------------- Run code --------------------------------------------------------

if __name__ == '__main__':

    # reorient MRI volumes:
    # convert_labels = ConvertLabels()

    # make folder tree:
    mft = MakeFolderTree()