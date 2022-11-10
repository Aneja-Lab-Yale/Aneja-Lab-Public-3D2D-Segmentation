# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# ----------------------------------------------- Imports -----------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import torch

# ----------------------------------------------- imshow function -----------------------------------------------

def imshow(img, voxsize=(1,1,1), coords=('L','A','S')):
    """
        Description:
            This function shows 2D/3D/4D MRI slice/volume/series:
            - 2D: image is shown.
            - 3D: the volume mid-slices in axial, coronal and sagittal planes are shown.
            - 4D: the series can be diffusion MRI (dMRI) or functional MRI (fMRI).
                The last component of the 4D data should represent series volumes (e.g. Z / Y / X / seriesVolume).
                It shows the first (b0) and the mid-series volumes.
                This function is optimized to show 4D diffusion MRI series, and assumes that the first volume is a b0 image.
                However, the function can show 4D fMRI data as well (first and mid-series volumes).
            - Torch batch: in case img is a torch batch, this function shows all images in the batch.

        Inputs:
            - img: 2D/3D/4D numpy array representing an MRI slice/volume/series
                    OR
                    Torch tensor representing a batch of 2D/3D/4D MRI slices/volumes/series.
            - voxsize: voxel size; 3-component tuple or list; default voxsize=(1,1,1)
            - coords: img coordinate system; 3-component character tuple or list; default coords=('L','I','A')

            In case img is a torch batch, voxsize and coords should describe the voxsize and coordinate system of
            each image in the batch.

        Output:
            None (plots img instead)

        Further info:
            You can use load_nifti (in dipy package) to obtain coords of an image, by setting return_coords=True.
            Example:
                from dipy.io.image import load_nifti
                img, affine, voxsize, coords = load_nifti('brainmask.mgz', return_voxsize=True, return_coords=True)
            To learn more about voxel coordinate systems, see:
                http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
        """

    # If img is a torch batch, show every image in the batch:
    if torch.is_tensor(img):
        show_torch_batch(img, voxsize, coords)
        # Note that voxsize and coords should describe the voxsize and coordinate system of each image in the batch.
        return


    dim = img.ndim
    if dim in (3, 4):
        img, voxsize = correct_coordinates(img, voxsize, coords)
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxsize[1] / voxsize[0]
        coronal_aspect_ratio = voxsize[2] / voxsize[0]
        sagittal_aspect_ratio = voxsize[2] / voxsize[1]


    kwargs = dict(cmap='gray', origin='lower')


    if dim == 2:
        plt.imshow(img.T, **kwargs)


    elif dim == 3 or (dim == 4 and img.shape[3] == 1):
        axial = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial, ...].T, **kwargs)
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :, ...].T, **kwargs)
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :, ...].T, **kwargs)
        sagittal.set_aspect(sagittal_aspect_ratio)


    elif dim == 4:
        midresies = img.shape[3] // 2

        axial_b0 = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial, 0].T, **kwargs)
        axial_b0.set_aspect(axial_aspect_ratio)

        coronal_b0 = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :, 0].T, **kwargs)
        coronal_b0.set_aspect(coronal_aspect_ratio)

        sagittal_b0 = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :, 0].T, **kwargs)
        sagittal_b0.set_aspect(sagittal_aspect_ratio)

        axial_midseries = plt.subplot(2, 3, 4)
        plt.imshow(img[:, :, midaxial, midresies].T, **kwargs)
        axial_midseries.set_aspect(axial_aspect_ratio)

        coronal_midseries = plt.subplot(2, 3, 5)
        plt.imshow(img[:, midcoronal, :, midresies].T, **kwargs)
        coronal_midseries.set_aspect(coronal_aspect_ratio)

        sagittal_midseries = plt.subplot(2, 3, 6)
        plt.imshow(img[midsagittal, :, :, midresies].T, **kwargs)
        sagittal_midseries.set_aspect(sagittal_aspect_ratio)

    plt.show()


# ----------------------------------------------- Helper functions -----------------------------------------------

def show_torch_batch(torch_batch, voxsize, coords):
    """
    This function shows all images in a torch batch.

    :param torch_batch: torch batch
    :param voxsize: voxel size of each image in the batch
    :param coords: coordinate system of each image in the batch
    """
    n = torch_batch.shape[0]  # n = number of images in Torch batch
    m = torch_batch.shape[1]  # m = number of image channels
    print(f'''
        -----------------------------------------
        ImageShow function alert: 
        showing {n} images in this pytorch batch.
        -----------------------------------------
        ''')
    for i in range(n):
        for j in range(m):
            image = torch_batch[i, j, ...].numpy()
            imshow(image, voxsize, coords)


def correct_coordinates(img, voxsize, coords):
    """
    This function re-orients a 3D MRI volume or 4D MRI series  into the standard radiology coordinate system:
    ('L','A','S').

    If img is a 4D MRI series, the last component of the 4D MRI series should represent series volumes
    (e.g. Z / Y / X / seriesVolume).

    If the image coordinate system was not identified when you called the function and you received this error:
        'coords not identified: please go to MRIslicer.imshow and define the coordinate system :)
    you should revise this sub-function!

    Please do the following steps:
    1. Find the image coordinate system by using these commands:

    from dipy.io.image import load_nifti
    img, affine, voxsize, coords = load_nifti("brainmask.mgz", return_voxsize=True, return_coords=True)

    2. Add the new coords by following these steps:
      a. Compared coords with the standard radiology system of coordinates: ('L','A','S').
      b. Use np.swapaxes and np.flip to transform img coordinates to standard radiology system.

    For more info about voxel coordinate systems, see:
    http://www.grahamwideman.com/gw/brain/orientation/orientterms.htm
    """
    dim = img.ndim

    # If img is a 4D MRI series, correct every volume in the series:
    if dim == 4:
        for i in range(img.shape[3]):
            img[..., i], vxsize = correct_coordinates(img[..., i], voxsize, coords)
        return img, voxsize

    if coords == ('L', 'A', 'S'):
        return img, voxsize
    if coords == ('R', 'A', 'S'):
        img = np.flip(img, 0)
        return img, voxsize
    if coords == ('L', 'I', 'A'):
        img = np.swapaxes(img, 1, 2)
        img = np.flip(img, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'S', 'A'):
        img = np.swapaxes(img, 1, 2)
        voxsize = (voxsize[0], voxsize[2], voxsize[1])
        return img, voxsize
    if coords == ('L', 'P', 'S'):
        img = np.flip(img, 1)
        return img, voxsize
    raise Exception('coords not identified: please go to MRIslicer.imshow and define the coordinate system :)')


# ----------------------------------------------- Test code -----------------------------------------------

if __name__ == "__main__":
    from dipy.io.image import load_nifti
    image, affine, voxsize, coords = load_nifti(
        '/Users/arman/projects/capsnet/data/images/003_S_1074/2008-08-19_14_11_12.0/brainbox.nii',
        return_voxsize=True, return_coords=True)

    imshow(image)
    imshow(image, voxsize, coords)