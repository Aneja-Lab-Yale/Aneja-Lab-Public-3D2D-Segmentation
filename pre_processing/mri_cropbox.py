# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)


# ---------------------------------------------------- Imports -----------------------------------------------------

import os
from os.path import join

import pandas as pd
from tqdm import tqdm
import numpy as np
from dipy.io.image import load_nifti, save_nifti

# -------------------------------------------------- MRICrop class -------------------------------------------------

class MRICrop:

    def __init__(self):
        ###########################################################
        #                  SET CROPPING PARAMETERS                #
        ###########################################################

        # Set the size of the cropped volume:
        # if this is set to 100, the center of the volumed is cropped with the size of 100 x 100 x 100.
        # if this is set to (100, 64, 64), the center of the volume is cropped with size of (100 x 64 x 64).
        # note that 100, 64 and 64 here respectively represent left-right, posterior-anterior,
        # and inferior-superior dimensions, i.e. standard radiology coordinate system ('L','A','S').
        self.crop = (64, 64, 64)
        # Set cropshift:
        # if the target structure is right hippocampus, the crop box may be shifted to right by 20 pixels,
        # anterior by 5 pixels, and inferior by 20 pixels --> cropshift = (-20, 5, -20);
        # note that crop and cropshift here are set here using standard radiology system (L, A, S):
        self.cropshift = (-20, 0, -20)

        # Project root:
        self.project_root = '/Users/sa936/projects/capsnet'
        # Images folder:
        self.images_folder = 'data/valid_images'
        # Files in which cropbox are put in:
        self.image_name = 'brainbox.mgz'

        # Cropboxes folder:
        self.cropbox_folder = 'data/valid_images'
        # Cropbox files:
        self.cropbox_name = 'cropbox.nii.gz'
        # Limit cropboxes to this number:

        # .........................................................................................................
        ###################################
        #   DON'T CHANGE THESE, PLEASE!   #
        ###################################
        self.image_list = make_files_paths(join(self.project_root, self.images_folder))
        self.image_list = filter_files_paths(self.image_list, self.image_name)

        self.crop_files()
        self.write_stats()

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


    def crop_files(self):
        for image_path in tqdm(self.image_list, desc='Making cropbox NIfTIs'):
            image, affine = load_nifti(image_path)
            # .....................................................................................................
            (x, z, y) = image.shape     # this works if inputs & outputs are in ('L','I','A') coordinate system
            xcrop, ycrop, zcrop = self.crop[0], self.crop[1], self.crop[2]
            xshift, yshift, zshift = self.cropshift[0], self.cropshift[1], - self.cropshift[2]  # ('L','I','A') system

            xmid, ymid, zmid = x // 2, y // 2, z // 2
            xdiff, ydiff, zdiff = xcrop // 2, ycrop // 2, zcrop // 2

            xstart, ystart, zstart = xmid - xdiff + xshift, ymid - ydiff + yshift, zmid - zdiff + zshift
            xstop, ystop, zstop = xmid + xdiff + xshift, ymid + ydiff + yshift, zmid + zdiff + zshift

            if xstart < 0:
                xstop -= xstart
                xstart = 0
            if ystart < 0:
                ystop -= ystart
                ystart = 0
            if zstart < 0:
                zstop -= zstart
                zstart = 0
            if xstop > x:
                xstart -= xstop - x
                xstop = x
            if ystop > y:
                ystart -= ystop - y
                ystop = y
            if zstop > z:
                zstart -= zstop - z
                zstop = z
            assert (xstart >= 0) and (ystart >= 0) and (zstart >= 0) and (xstop <= x) and (ystop <= y) and (zstop <= z)
            # .....................................................................................................
            box = np.zeros_like(image).astype('uint8')
            box[xstart, zstart:zstop, ystart:ystop] = box[xstop, zstart:zstop, ystart:ystop] = \
                box[xstart:xstop, zstart, ystart:ystop] = box[xstart:xstop, zstop, ystart:ystop] = \
                box[xstart:xstop, zstart:zstop, ystart] = box[xstart:xstop, zstart:zstop, ystop] = 1
            # .................................................................................................
            '''
            Example of a path:
            /Users/arman/projects/capsnet/data/images/033_S_0725/2008-08-06_13_54_42.0/aparc+aseg_brainbox.mgz
            '''
            path_components = image_path.split('/')
            subject, scan = path_components[-3], path_components[-2]
            folder = join(self.project_root, self.cropbox_folder, subject, scan)
            os.makedirs(folder, exist_ok=True)

            save_nifti(join(folder, self.cropbox_name), box, affine)

    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    def write_stats(self):

        cropping_parameters = pd.DataFrame(index=['crop size',
                                                  'crop shift',
                                                  'input image'],
                                           data=[str(self.crop),
                                                 str(self.cropshift),
                                                 str(self.image_name)])

        cropping_parameters.to_csv(join(self.project_root, self.cropbox_folder, 'crop_parameters.csv'),
                                   header=False)


# ---------------------------------------------- Helper functions -------------------------------------------------


def make_files_paths(folder):
    """
    :return: paths to all files in the folder passed to function
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



# ----------------------------------------------- Run MRICrop instance --------------------------------------------

# Run cropping:
mri_crop = MRICrop()


