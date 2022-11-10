# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# -------------------------------------------------- Imports --------------------------------------------------------

import os
from os.path import join
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import numpy as np
import pandas as pd

from dipy.io.image import load_nifti, save_nifti



# --------------------------------------------- MRIReorient class ---------------------------------------------------

class MRIReorient:
    """
    This class re-orients the MRI volumes from any coordinate system into the standard radiology system,
    i.e. ('L','A','S') coordinate system.
    """
    def __init__(self):
        #####################################################
        # SET THESE:
        self.project_root = '/Users/sa936/projects/capsnet'
        self.images_csv = 'data/datasets_local/combined.csv'
        self.corrected_folder = 'data/images_corrected'
        #####################################################

        self.image_list = make_image_list(join(self.project_root, self.images_csv))

        for image_path in tqdm(self.image_list, desc='Correcting MRI orientations'):
            image, affine, coords = load_nifti(image_path, return_coords=True)
            image, affine = correct_coordinates(image, affine, coords)

            folder, file = make_corrected_paths(image_path, self.project_root, self.corrected_folder)

            os.makedirs(folder, exist_ok=True)
            save_nifti(join(folder, file), image, affine)



# ----------------------------------------------- MRIOrient class ----------------------------------------------------

class MRIOrient:
    """
    This class scans the entire input MRI volumes and output MRI masks to ensure that all inputs and outputs
    are in the same coordinate system.
    """
    def __init__(self):
        """
        Notes:
            - The csv files should not have headers or indexing.
            - If you make the csv files with subjects_train_test_split.py, the file format will be correct.
        """
        #####################################################
        # SET THESE:
        self.project_root = '/Users/sa936/projects/capsnet'
        self.coorected_folder = 'data/images_corrected'
        #####################################################

        self.image_list = make_files_list(join(self.project_root, self.coorected_folder))
        self.image_list = filter_files(self.image_list)

        self.coords_list = make_coords_list(self.image_list)

        self.unique_coords = set(self.coords_list)

        print(f'Unique coordinates in the entire MRI volumes: \n'
              f'{self.unique_coords} \n')


# ---------------------------------------------- Helper functions ---------------------------------------------------

def correct_coordinates(image, affine, coords):
    """
    This function re-orients the MRI volume into the standard radiology system, i.e. ('L','A','S'),
    and also corrects the affine transform for the volume (from the MRI volume space to the scanner space).

    Inputs:
        - image: MRI volume
        - affine: affine transform for the MRI volume
        - coords: the coordinate system of the MRI volume.

    Outputs:
        - image: corrected MRI volume in the standard radiology coordinate system.
        - affine: corrected affine transform (from MRI volume space to the scanner space).
    """
    if coords == ('L', 'A', 'S'):
        return image, affine

    if coords == ('R', 'A', 'S'):
        image = np.flip(image, 0)
        affine[:, 0] = - affine[:, 0]
        return image, affine

    if coords == ('L', 'S', 'A'):
        image = np.swapaxes(image, 1, 2)
        affine[:, [1, 2]] = affine[:, [2, 1]]
        return image, affine

    if coords == ('L', 'I', 'A'):
        # L,I,A --> L,A,I
        image = np.swapaxes(image, 1, 2)
        affine[:, [1, 2]] = affine[:, [2, 1]]
        # L,A,I --> L,A,S
        image = np.flip(image, 2)
        affine[:, 2] = - affine[:, 2]
        return image, affine
    if coords == ('L', 'P', 'S'):
        image = np.flip(image, 1)
        affine[:, 1] = - affine[:, 1]
        return image, affine

    raise Exception('coords not identified: please go to pre_processing.MRIorient and define the coordinate system'
                    'within correct_coordinates function.')



def make_coords_list(image_list):
    """
    Input:
        - image_list: list of paths to MRI volumes

    Output:
        - coords_list: list of coordinates of all MRI volumes in the list
    """
    # Parallel processing:
    # coords_list = process_map(compute_coords, image_list)

    # Non-parallel processing:
    coords_list = []
    for image_path in tqdm(image_list, desc='Computing coordinates of corrected MRIs'):
        coords = get_coords(image_path)
        coords_list.append(coords)

    return coords_list



def get_coords(image_path):
    """
    Input:
        - image_path: path to a single MRI volume

    Output:
        - coordinates of the MRI volume, e.g. ('L','A','S')
    """
    _, _, coords = load_nifti(image_path, return_coords=True)
    """
    This is the complete format for reference:
    image, affine, voxsize, coords = load_nifti(image_path, return_voxsize=True, return_coords=True)
    """
    return coords



def make_image_list(images_csv):
    images_df = pd.read_csv(images_csv, header=None)
    image_list = list(images_df.iloc[:, 0])
    return image_list



def make_corrected_paths(image_path, project_root, corrected_folder):
    """
    Example path
    /Users/arman/projects/capsnet/data/images/137_S_0825/2007-10-29_12_12_41.0/aparc+aseg_brainbox.mgz
    """
    path_components = image_path.split('/')
    subject, scan, file = path_components[-3], path_components[-2], path_components[-1]
    file = file[:-3] + 'nii.gz'                                                        # replace .mgz with .nii.gz
    folder = join(project_root, corrected_folder, subject, scan)
    return folder, file



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



def filter_files(file_list, structure='.nii'):
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
    mrireorient = MRIReorient()

    # check results:
    mriorient = MRIOrient()
    """
    Expected printed results:
    
    Unique coordinates in the entire MRI volumes:    
    {('L', 'A', 'S')}
    """

