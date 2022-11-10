# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# ----------------------------------- Imports -----------------------------------

import os
from os.path import join
from tqdm.contrib.concurrent import process_map

# ----------------------------------- Create Brain Bounding Box class -----------------------------------

class CreateBrainBoundingBox:
    """
    This class:
    1. makes binary masks for the brain out of aparc+aseg.mgz files.
    2. Puts a bounding box around the brain images and crops it.
    3. Puts a bounding box around any output file (e.g. aparc+aseg.mgz) and crops it.

    FreeSurfer output files (e.g. aparc+aseg.mgz) should be organized in this way:

    > images / subjects / scans / aparc+aseg.mgz

    Input brain images should be organized in this way:

    > images / subjects / scans / brainmask.mgz

    This will put brainmask_binary.mgz, brainbox.mgz, and cropped output files (e.g. aparc+aseg_brainbox.mgz)
    in the image folders.

    Because the run_commands method uses multiprocessing (see below), it extracts the masks much faster
    as compared to running the commands in terminal or in a for loop.
    """

    def __init__(self,
                 make_brainmask_binary=True,
                 make_brainimage_box=True,
                 make_outputs_boxes=True):

        ###########################################################
        #         SET BOUNDING BOX MAKER PARAMETERS HERE!         #
        ###########################################################

        # Set project root:
        self.project_root = '/Users/arman/projects/capsnet'

        # Set the folders:
        self.images_folder = 'data/images'

        # Set the file names:
        self.brainimage = 'brainmask.mgz'
        self.brainmask_binary = 'brainmask_binary.mgz'
        self.brainbox = 'brainbox.mgz'
        self.aparc_aseg = 'aparc+aseg.mgz'

        ###################################################################################

        if make_brainmask_binary:
            self.brainmask_binary_commands = self.make_brainmask_binary_commands()
            run_parallel(self.brainmask_binary_commands)

        if make_brainimage_box:
            self.brainibox_commands = self.make_brainbox_commands()
            run_parallel(self.brainibox_commands)

        if make_outputs_boxes:
            self.outputs_brainbox_commands = self.make_outputs_brainbox_commands()
            run_parallel(self.outputs_brainbox_commands)

        ###################################################################################

    def make_brainmask_binary_commands(self):
        # Make a list of input images (brainmask.mgz) paths:
        brainimages_temp = make_files_paths(join(self.project_root, self.images_folder))
        brainimages = filter_files_paths(brainimages_temp, self.brainimage)
        # Make a list of binary brainmasks (brainmask_binary) paths:
        brainmasks_binary = paths_replace(brainimages, self.brainimage, self.brainmask_binary)
        # Make a list of aparc+aseg.mgz paths:
        aparc_asegs_temp = make_files_paths(join(self.project_root, self.images_folder))
        aparc_asegs = filter_files_paths(aparc_asegs_temp, self.aparc_aseg)

        # Make the list of commands to extract binary brain masks from aparc+aseg.mgz files;
        # the generated files (brainmask_binary.mgz) are put in corresponding inputs folders:
        brainmask_binary_commands = []

        for i in range(len(brainimages)):
            brainmask_binary_commands.append(f'mri_binarize --i {aparc_asegs[i]} '
                                             f'--o {brainmasks_binary[i]} --min 0.5')

        return brainmask_binary_commands



    def make_brainbox_commands(self):
        # Make a slit of input images (brainmask.mgz) paths:
        brainimages_temp = make_files_paths(join(self.project_root, self.inputs_folder))
        brainimages = filter_files_paths(brainimages_temp, self.brainimage)
        # Make a list of binary brainmasks (brainmask_binary.mgz) paths:
        brainmasks_binary = paths_replace(brainimages, self.brainimage, self.brainmask_binary)
        # Maek a list of cropped brainbox image (brainbox.mgz) paths:
        brainboxes = paths_replace(brainimages, self.brainimage, self.brainbox)

        # Make the list of commands to extract brain boxe images (brainbox.mgz)
        # from input brain images (brainmask.mgz) using binary brain masks (brainmask_binary.mgz);
        # the generated files (brainbox.mgz) are put in corresponding inputs folders:
        brainbox_commands = []

        for i in range(len(brainimages)):
            brainbox_commands.append(f'mri_mask -bb 1 {brainimages[i]} {brainmasks_binary[i]} {brainboxes[i]}')

        return brainbox_commands



    def make_outputs_brainbox_commands(self):
        # Make a list of binary brain masks (brainmask_binary.mgz) paths:
        brainmasks_binary_temp = make_files_paths(join(self.project_root, self.inputs_folder))
        brainmasks_binary = filter_files_paths(brainmasks_binary_temp, self.brainmask_binary)

        # Outputs: 2-dimensional list of output images:
        #   dimension 1: n output structures (e.g. aparc+aseg.mgz, l_hippocampus.mgz, ...); length: n_structures
        #   dimension 2: list of files paths in all subjects to that output structure; length: n_subjects
        outputs_temp = make_files_paths(join(self.project_root, self.outputs_folder))
        outputs = []
        for output_name in self.outputs:
            outputs.append(filter_files_paths(outputs_temp, output_name))

        n_structures = len(outputs)
        assert len(brainmasks_binary) == len(outputs[0])
        n_subjects = len(brainmasks_binary)

        # Outputs_brainboxes: 2-dimensional list with same structure as outputs:
        # in this 2-dimensional list, the cropped output image paths are generated;
        # e.g. path to aparc+aseg.mgz --> path to aparc+aseg_brainbox.mgz
        outputs_brainboxes = []
        for i in range(n_structures):
            outputs_brainboxes.append(paths_replace(outputs[i], self.outputs[i], self.outputs_brainboxes[i]))

        # Make 1-dimensional list of all commands to put a brainbox around output images and crop them:
        outputs_brainbox_commands = []
        for i in range(n_structures):
            for j in range(n_subjects):
                outputs_brainbox_commands.append(f'mri_mask -bb 1 '
                                                 f'{outputs[i][j]} '                # output i in subject j
                                                 f'{brainmasks_binary[j]} '         # brainmask_binary in subject j 
                                                 f'{outputs_brainboxes[i][j]}')     # cropped output i in subject j

        return outputs_brainbox_commands



def make_files_paths(folder):
    """
    Output:
    - file_paths: paths to all files in the folder passed to function
    """

    files_paths = []

    for path, _, files in os.walk(folder):
        for file in files:
            files_paths.append(join(path, file))

    return files_paths



def filter_files_paths(path_list, name):
    """
    This function takes in a list of paths,
    and returns the list of paths that contain "name".
    """
    return [file_path for file_path in path_list if name in file_path]



def paths_replace(path_list, str1, str2):
    """
    Inputs:
        - path_list: list of paths to files
        - str1: string to be crossed out in each element of paths_list
        - str2: string to be added to each element of paths_list

    Output:
        - corrected path_list with elements that have undergone replacement str1 --> str2
    """
    return [path.replace(str1, str2) for path in path_list]



def run_parallel(commands):
    """
    This  function parallel-runs the commands in Terminal using all cpu cores.
    """
    process_map(os.system, commands)


# ----------------------------------- Generate masks -----------------------------------

# Initiate instance:
bbox = CreateBrainBoundingBox()
