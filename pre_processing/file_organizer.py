# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# ----------------------------------- Imports -----------------------------------

import os
from os.path import join
from tqdm import trange
from tqdm.contrib.concurrent import process_map
import pandas as pd

# ----------------------------------- Move Files Class -----------------------------------

class MoveFiles:
    """
    This class moves target files from the inputs folder to outputs folder.
    """
    def __init__(self):
        self.project_root = '/Users/arman/projects/capsnet'
        self.inputs_folder = 'data/brainmasks'
        self.outputs_folder = 'data/fs_segmentations'
        self.target_files = ['aparc+aseg.mgz', 'aparc+aseg_brainbox.mgz']

        self.files_paths = make_files_paths(join(self.project_root, self.outputs_folder))
        self.files_paths = filter_files_paths(self.files_paths, self.target_files)

        self.commands = self.make_commands()
        run_parallel(self.commands)


    def make_commands(self):
        from_list = self.files_paths
        to_list = [path.replace(self.outputs_folder, self.inputs_folder) for path in from_list]
        commands = []
        for i in trange(len(from_list), desc='Making move commands'):
            commands.append(f'mv {from_list[i]} {to_list[i]}')
        return commands

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

class CopyDatasetImages:
    """
    This class moves target files from the inputs folder to outputs folder.
    """
    def __init__(self):

        self.project_root = '/Users/Emad/projects/capsnet'
        self.dataset_folder = 'data/datasets_local'
        self.dataset_images = 'valid_outputs.csv'

        self.tofolder = 'data/valid_images'
        # .....................................................................................................
        self.files_paths = make_image_list(join(self.project_root, self.dataset_folder, self.dataset_images))
        self.copy_files()


    def copy_files(self):
        for path in self.files_paths:
            '''
            Example of a path:
            /home/arman_avesta/capsnet/data/images/033_S_0725/2008-08-06_13_54_42.0/brainbox.mgz
            '''
            path_components = path.split('/')
            subject, scan = path_components[-3], path_components[-2]
            folder = join(self.project_root, self.tofolder, subject, scan)
            os.makedirs(folder, exist_ok=True)
            os.system(f'cp {path} {folder}')

# ------------------------------------------------- Helper functions -------------------------------------------------

def make_image_list(path_to_images_csv):
    image_df = pd.read_csv(path_to_images_csv, header=None)
    image_list = list(image_df.iloc[:, 0])
    return image_list


def make_files_paths(root):
    files_paths = []
    for path, _, files in os.walk(root):
        for file in files:
            files_paths.append(join(path, file))
    return files_paths


def filter_files_paths(files_paths, target_files):
    filtered_files_paths = []
    for target_file in target_files:
        for file_path in files_paths:
            if target_file in file_path:
                filtered_files_paths.append(file_path)
    return filtered_files_paths


def run_parallel(commands):
    """
    This  function parallel-runs the the commands in Terminal using all cpu cores.
    """
    process_map(os.system, commands)


# ----------------------------------- Move Files -----------------------------------

cdi = CopyDatasetImages()