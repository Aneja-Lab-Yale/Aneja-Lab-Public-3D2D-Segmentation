# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# ----------------------------------- Imports -----------------------------------

import os
from os.path import join
import numpy as np
import pandas as pd

# ----------------------------------- SubjectsTestTrainSplit class -----------------------------------

class SubjectsTrainValidTestSplit:
    """
    This class splits the data into training, validation and test sets AT THE SUBJECT LEVEL:

    assume that you have matching folders for inputs (e.g. brain MRIs) and outputs (e.g. hippocampus masks),
    but organized as:

    > inputs root / subjects / scans / brain.nifti

    > outputs root / subjects / scans / hippocampus.nifti

    Each subject may have multiple follow-up scans.
    In this paradigm, the split into training, validation  and test sets should be done at the subject level,
    not at the image level:
    if we split at the image level, a scan of one subject may end up in the training set, and another scan of the
    same subject may end up in the test set. Since these two scans are correlated, this will inflate the performance
    of the model during testing.

    This class first puts aside n and m subjects as the validation and test sets.
    Then, it generates paths to all input and ouput files, separately for the subjects in the train, valid & test sets.
    Finally, it generates 7 csv files:
    - train_inputs: list of paths to input files in the training set
    - valid_inputs: list of paths to input files in the validation set
    - test_inputs: list of paths to input files in the test set
    - train_outputs: list of paths to output files in the training set
    - valid outputs: list of paths to output files in the validation set
    - test_outputs: list of paths to output files in the test set
    - train_valid_test_split_summary: contains the summary of train/test split numbers (subjects and images).

    Please set the class parameters in the __init__ below.
    """

    def __init__(self):

        #############################################################
        #                  SET SPLIT PARAMETERS HERE!               #
        #############################################################

        # Set the size of validation set:
        # if valid_size < 1 --> proportion of subjects to be assigned to validation
        # if valid_size > 1 --> number of subjects to be assigned to validation
        self.valid_size = 30

        # Set the size of test set:
        self.test_size = 30

        self.project_root = '/Users/arman/projects/capsnet'

        # Set the root of the inputs directory:
        self.inputs_folder = 'data/brainmasks'

        # Set the input file name:
        # the input files in all folders should have the same name (brainmask.mgz here).
        # if your input files have different names in different folders, use file_organizer.py to rename & organize
        # your inputs.
        self.input_structure = 'brainmask.mgz'

        # Set the root of the outputs directory:
        self.outputs_folder = 'data/fs_segmentations'

        # Set the output file name:
        # similar to inputs, the outputs in all folders should have the same name.
        self.output_structure = 'third_ventricle.mgz'  # Here, the segmentation target is third ventricle.

        # Set the directory in which the 5 files created by this class will be saved.
        self.info_dir = 'data/info'

        ####################################################################

        self.inputs_root = join(self.project_root, self.inputs_folder)
        self.outputs_root = join(self.project_root, self.outputs_folder)

        (self.train_subjects,
         self.valid_subjects,
         self.test_subjects) = self.split_subjects()

        (self.train_inputs,
         self.valid_inputs,
         self.test_inputs,
         self.train_outputs,
         self.valid_outputs,
         self.test_outputs) = self.create_image_lists()

        print(f'''
        Total number of subjects:       {self.total_subjects()}
        Number of training subjects:    {len(self.train_subjects)} 
        Number of validation subjects:  {len(self.valid_subjects)}
        Number of test subjects:        {len(self.test_subjects)} 
        
        Total number of images:         {self.total_images()} 
        Number of training images:      {len(self.train_outputs)} 
        Number of validation images:    {len(self.valid_outputs)}
        Number of test images:          {len(self.test_outputs)}
        ''')

        self.write_files()

        ####################################################################

    def split_subjects(self):

        assert os.listdir(self.inputs_root) == os.listdir(self.outputs_root), \
            'The input and output folder trees should match!'

        subjects = os.listdir(self.inputs_root)

        n = len(subjects)

        self.valid_size = round(n * self.valid_size) if self.valid_size < 1 else self.valid_size
        self.test_size = round(n * self.test_size) if self.test_size < 1 else self.test_size

        indexes = list(range(n))
        np.random.shuffle(indexes)

        valid_indexes = indexes[: self.valid_size]
        test_indexes = indexes[self.valid_size: self.valid_size + self.test_size]
        train_indexes = indexes[self.valid_size + self.test_size:]


        train_subjects = [subjects[index] for index in train_indexes]
        valid_subjects = [subjects[index] for index in valid_indexes]
        test_subjects = [subjects[index] for index in test_indexes]

        return train_subjects, valid_subjects, test_subjects



    def create_image_lists(self):
        """
        :return: list of paths to inputs and outputs for training, validation and test sets.
        """
        train_inputs_temp = make_file_list(self.inputs_root, self.train_subjects)
        valid_inputs_temp = make_file_list(self.inputs_root, self.valid_subjects)
        test_inputs_temp = make_file_list(self.inputs_root, self.test_subjects)
        train_outputs_temp = make_file_list(self.outputs_root, self.train_subjects)
        valid_outputs_temp = make_file_list(self.outputs_root, self.valid_subjects)
        test_outputs_temp = make_file_list(self.outputs_root, self.test_subjects)

        train_inputs = filter_files(train_inputs_temp, self.input_structure)
        valid_inputs = filter_files(valid_inputs_temp, self.input_structure)
        test_inputs = filter_files(test_inputs_temp, self.input_structure)
        train_outputs = filter_files(train_outputs_temp, self.output_structure)
        valid_outputs = filter_files(valid_outputs_temp, self.output_structure)
        test_outputs = filter_files(test_outputs_temp, self.output_structure)

        return train_inputs, valid_inputs, test_inputs, train_outputs, valid_outputs, test_outputs



    def total_subjects(self):
        """
        :return: total number of subjects
        """
        return len(self.train_subjects) + len(self.valid_subjects) + len(self.test_subjects)



    def total_images(self):
        """
        :return: total number of images
        """
        # Returns the number of images (here number of MRI volumes) for all subjects
        return len(self.train_inputs) + len(self.valid_inputs) + len(self.test_inputs)



    def write_files(self):
        """
        This method writes csv files containing subject lists. It also writes a summary file.
        """
        train_inputs_df = pd.DataFrame(self.train_inputs)
        valid_inputs_df = pd.DataFrame(self.valid_inputs)
        test_inputs_df = pd.DataFrame(self.test_inputs)
        train_outputs_df = pd.DataFrame(self.train_outputs)
        valid_outputs_df = pd.DataFrame(self.valid_outputs)
        test_outputs_df = pd.DataFrame(self.test_outputs)
        summary_df = pd.DataFrame(index=['Total number of subjects',
                                         'Number of training subjects',
                                         'Number of validation subjects',
                                         'Number of test subjects',
                                         'Total number of images',
                                         'Number of training images',
                                         'Number of validation images',
                                         'Number of test images'],
                                  data=[self.total_subjects(),
                                        len(self.train_subjects),
                                        len(self.valid_subjects),
                                        len(self.test_subjects),
                                        self.total_images(),
                                        len(self.train_outputs),
                                        len(self.valid_outputs),
                                        len(self.test_outputs)])

        train_inputs_path = join(self.project_root, self.info_dir, 'train_inputs.csv')
        valid_inputs_path = join(self.project_root, self.info_dir, 'valid_inputs.csv')
        test_inputs_path = join(self.project_root, self.info_dir, 'test_inputs.csv')
        train_outputs_path = join(self.project_root, self.info_dir, 'train_outputs.csv')
        valid_outputs_path = join(self.project_root, self.info_dir, 'valid_outputs.csv')
        test_outputs_path = join(self.project_root, self.info_dir, 'test_outputs.csv')
        summary_path = join(self.project_root, self.info_dir, 'train_valid_test_split_summary.csv')

        kwargs = dict(header=False, index=False)
        train_inputs_df.to_csv(train_inputs_path, **kwargs)
        valid_inputs_df.to_csv(valid_inputs_path, **kwargs)
        test_inputs_df.to_csv(test_inputs_path, **kwargs)
        train_outputs_df.to_csv(train_outputs_path, **kwargs)
        valid_outputs_df.to_csv(valid_outputs_path, **kwargs)
        test_outputs_df.to_csv(test_outputs_path, **kwargs)
        summary_df.to_csv(summary_path, header=False)
        
        

# ------------------------------------------------ Helper functions ----------------------------------------------

        
def make_file_list(root, subjects):
    """
    :param root: root to either inputs or outputs folder
    :param subjects: list of train or validation or test subjects
    :return: paths to all files in the inputs or outputs folder of the subjects passed to function
    """
    file_list = []
    for subject in subjects:
        for path, _, files in os.walk(join(root, subject)):
            for file in files:
                file_list.append(join(root, path, file))
    return file_list


def filter_files(file_list, structure):
    """
    :param file_list: list of file paths
    :param structure: structure of interest: file paths will be filtered to contain this structure
    :return: filtered file list: only paths that contain the structure are returned
    """
    return [file_name for file_name in file_list if structure in file_name]



# ----------------------------------- Split subjects into train & test sets -----------------------------------

# Initiate split instance:
split = SubjectsTrainValidTestSplit()
