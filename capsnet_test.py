# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# -------------------------------------------------- Imports --------------------------------------------------

# Project imports:

from data_loader import AdniDataset, make_image_list
from capsnet_model_3d import CapsNet3D
from loss_functions import DiceLoss

# System imports:

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from os.path import join
from shutil import copyfile
from datetime import datetime
from tqdm import tqdm, trange
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dipy.io.image import save_nifti


# ---------------------------------------------- TestUNet3D class ----------------------------------------------

class TestCapsNet3D:

    def __init__(self, saved_model_path=None):
        self.start_time = datetime.now()

        ##########################################################
        #                  SET TESTING PARAMETERS                #
        ##########################################################

        # Set segmentation target:
        self.output_structure = 'right hippocampus'
        # Set FreeSurfer code for segmentation target:
        # to find the code, open any aparc+aseg.mgz in FreeView and change color coding to lookup table
        self.output_code = 53

        # Set the size of the cropped volume:
        # if this is set to 100, the center of the volumed is cropped with the size of 100 x 100 x 100.
        # if this is set to (100, 64, 64), the center of the volume is cropped with size of (100 x 64 x 64).
        # note that 100, 64 and 64 here respectively represent left-right, posterior-anterior,
        # and inferior-superior dimensions, i.e. standard radiology coordinate system ('L','A','S').
        self.crop = (64, 64, 64)
        # Set cropshift:
        # if the target structure is right hippocampus, the crop box may be shifted to right by 20 pixels,
        # anterior by 5 pixels, and inferior by 20 pixels --> cropshift = (-20, 5, -20);
        # note that crop and cropshift here are set here using standard radiology system ('L','A','S'):
        self.cropshift = (-20, 0, -20)

        # Set loss function: options are DiceLoss, criterion, and IoULoss:
        self.criterion = DiceLoss(conversion='threshold', reduction='none')

        # Project root:
        self.project_root = '/home/arman_avesta/capsnet'

        # Saved model paths:
        self.saved_model_folder = 'data/results/temp'
        self.saved_model_filename = 'saved_model.pth.tar'

        # Testing dataset paths:
        self.datasets_folder = 'data/datasets'
        # Testing on validation or test set:
        self.set = 'validation set'
        # csv file containing list of inputs for testing:
        self.test_inputs_csv = 'valid_inputs.csv'
        # csv file containing list of outputs for testing:
        self.test_outputs_csv = 'valid_outputs.csv'
        # csv file to which testing losses will be saved:

        # Set batch size (upper limit is determined by GPU memory):
        self.batch_size = 16

        # Set model: UNet3D
        self.model = CapsNet3D()

        # .......................................................................................................

        # Folder to save results and nifti files:
        # (cropped inputs, model predictions, and ground truth together with individual scan losses)
        self.niftis_folder = 'data/results/temp/niftis'
        # csv file to which testing hyperparameters and results will be saved:
        self.hyperparameters_file = 'testing_hyperparameters.csv'

        # Determine if backup to S3 should be done:
        self.s3backup = True
        # S3 bucket backup folder for results:
        self.s3_niftis_folder = 's3://aneja-lab-capsnet/data/results/temp/niftis'

        # .......................................................................................................
        ###################################
        #   DON'T CHANGE THESE, PLEASE!   #
        ###################################

        # Load model:
        self.saved_model_path = (join(self.project_root, self.saved_model_folder, self.saved_model_filename)
                                 if saved_model_path is None else saved_model_path)
        self.load_model()

        # Load testing dataset:
        self.inputs_paths = make_image_list(join(self.project_root, self.datasets_folder,
                                                 self.test_inputs_csv))
        self.outputs_paths = make_image_list(join(self.project_root, self.datasets_folder,
                                                  self.test_outputs_csv))
        self.dataset = AdniDataset(self.inputs_paths, self.outputs_paths, maskcode=self.output_code,
                                   crop=self.crop, cropshift=self.cropshift, testmode=True)
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)

        # Losses dataframe:
        self.losses = pd.DataFrame(columns=['subject', 'scan', 'loss type', 'loss'])

        # Run testing:
        self.test()

        # Save testing stats:
        self.save_stats()

        # Backup results to S3 bucket:
        if self.s3backup:
            self.backup_to_s3()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def test(self):
        print(f'''
        ###########################################################################
                            >>>   Starting testing   <<< 
        Saved model to be tested:               {self.saved_model_filename}        
        Segmentation target:                    {self.output_structure}
        Segmentation target code:               {self.output_code}
        Cropped image size:                     {self.crop}
        Crop shift in (L,A,S) system:           {self.cropshift}
        Number of examples:                     {len(self.dataset)}
        Number of batches:                      {len(self.dataloader)}
        Batch size:                             {self.batch_size}

        S3 folder:                              {self.s3_niftis_folder}
        ###########################################################################
        ''')
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        data_batches = tqdm(self.dataloader, desc='Testing')
        for data_batch in data_batches:

            inputs, targets, shapes, crops_coords, affines, paths = data_batch
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                outputs = self.model(inputs)
                batch_losses = self.criterion(outputs, targets)

            # .....................................................................................................

            for i in range(len(paths)):
                output = outputs[i, 0, ...].cpu().numpy()
                target = targets[i, 0, ...].cpu().numpy().astype('uint8')
                loss = batch_losses[i].cpu().numpy()
                shape = shapes[i, ...].numpy()
                cc = crops_coords[i, ...].numpy()  # cc: crop coordinates
                affine = affines[i, ...].numpy()
                path = paths[i]
                # .................................................................................................
                output_nc = np.zeros(shape)  # nc: non-cropped
                output_nc[cc[0, 0]: cc[0, 1], cc[1, 0]: cc[1, 1], cc[2, 0]: cc[2, 1]] = output

                target[0, :, :] = target[-1, :, :] = \
                    target[:, 0, :] = target[:, -1, :] = \
                    target[:, :, 0] = target[:, :, -1] = 1  # mark edges of the crop box

                target_nc = np.zeros(shape)
                target_nc[cc[0, 0]: cc[0, 1], cc[1, 0]: cc[1, 1], cc[2, 0]: cc[2, 1]] = target
                # .................................................................................................
                '''
                Example of a path:
                /home/arman_avesta/capsnet/data/images/033_S_0725/2008-08-06_13_54_42.0/aparc+aseg_brainbox.mgz
                '''
                path_components = path.split('/')
                subject, scan = path_components[-3], path_components[-2]
                folder = join(self.project_root, self.niftis_folder, subject, scan)
                os.makedirs(folder, exist_ok=True)

                save_nifti(join(folder, 'output.nii.gz'), output_nc, affine)
                save_nifti(join(folder, 'target.nii.gz'), target_nc, affine)
                # .................................................................................................
                scan_loss = pd.DataFrame({'subject': subject, 'scan': scan,
                                          'loss type': self.criterion, 'loss': [loss]})
                scan_loss.to_csv(join(folder, 'loss.csv'), index=False)

                self.losses = pd.concat([self.losses, scan_loss])

            # .....................................................................................................

            data_batches.set_description(f'Testing (loss: {self.losses["loss"].mean(): .3f}')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def save_stats(self):
        """
        This function writes the testing results onto csv files.
        Outputs:
            - testing_losses.csv
            - testing_times.csv (computation times)
            - testing_hyperparameters.csv
            These files will be saved in the path set by self.results_folder
        """
        os.makedirs(join(self.project_root, self.niftis_folder), exist_ok=True)
        computation_time = datetime.now() - self.start_time

        hyperparameters = pd.DataFrame(index=['date and time',
                                              'segmentation target',
                                              'freesurfer code for segmentation target',
                                              'image crop size',
                                              'crop shift in (L,A,S) system',
                                              '-----------------------------------------------',
                                              'testing on:',
                                              'number of examples',
                                              'batch size',
                                              '-----------------------------------------------',
                                              'total computation time',
                                              'computation time per example',
                                              '-----------------------------------------------',
                                              'loss function',
                                              'loss',
                                              '-----------------------------------------------',
                                              'model',
                                              '-----------------------------------------------',
                                              'S3 NIfTIs folder',
                                              'inputs',
                                              'outputs'],
                                       data=[datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                                             self.output_structure,
                                             self.output_code,
                                             self.crop,
                                             self.cropshift,
                                             '-----------------------------------------------',
                                             self.set,
                                             len(self.dataset),
                                             self.batch_size,
                                             '-----------------------------------------------',
                                             computation_time,
                                             computation_time / len(self.dataset),
                                             '-----------------------------------------------',
                                             self.criterion,
                                             self.losses['loss'].mean(),
                                             '-----------------------------------------------',
                                             self.model,
                                             '-----------------------------------------------',
                                             self.s3_niftis_folder,
                                             self.inputs_paths,
                                             self.outputs_paths])

        hyperparameters.to_csv(join(self.project_root, self.niftis_folder,
                                    self.hyperparameters_file), header=False)
        self.losses.to_csv(join(self.project_root, self.niftis_folder,
                                'scans_losses.csv'), index=False)

        print(f">>>   Testing loss: {self.losses['loss'].mean(): .3f}   <<<")

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def load_model(self):
        checkpoint = torch.load(self.saved_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        print(f'>>>   Loaded the model from: {self.saved_model_path}   <<<')

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def backup_to_s3(self, verbose=False):
        """
        This method backs up the results to S3 bucket.
        """
        ec2_folder = join(self.project_root, self.niftis_folder)
        command = f'aws s3 sync {ec2_folder} {self.s3_niftis_folder}' if verbose \
            else f'aws s3 sync {ec2_folder} {self.s3_niftis_folder} >/dev/null &'

        os.system(command)
        print(f'>>>   S3 backup done   <<<')


# ------------------------------------------ Run TrainUNet3D Instance ------------------------------------------

# Test the network:
utest = TestCapsNet3D()
