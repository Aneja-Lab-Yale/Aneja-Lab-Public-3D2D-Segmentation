# 2D 2.5D 3D Segmentation Project
# Aneja Lab | Yale School of Medicine
# Developed by Arman Avesta, MD
# Created (3/30/22)
# Updated (11/5/22)

# ------------------------------------------------- ENVIRONMENT SETUP -------------------------------------------------

# Project imports:


# System imports:
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

# Print configs:
np.set_printoptions(precision=1, suppress=True)
torch.set_printoptions(precision=1, sci_mode=False)


# ------------------------------------------------ HELPER FUNCTIONS ---------------------------------------------------

def imshow(img, voxsize=(1,1,1), coords=('L','A','S')):
    """
        Description:
            This function shows 2D/3D/4D MRI slice/volume:
            - 2D: image is shown.
            - 3D: the volume mid-slices in axial, coronal and sagittal planes are shown.
            - 4D: assumes that image is multi-channel image (channel-first) and shows all channels as 3D channels.
            - 5D: assumes that image is a batch of multi-channel images (batch first, channel second), and shows all
                all batches & all channels of 3D images.
        Inputs:
            - img: 2D/3D numpy array representing an MRI slice/volume/series
                    OR
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
    kwargs = dict(cmap='gray', origin='lower')
    ndim = img.ndim
    assert ndim in (2, 3, 4 ,5), f'image shape: {img.shape}; imshow can only show 2D and 3D images, ' \
                                 f'multi-channel 3D images (4D), and batches of multi-channel 3D images (5D).'

    if ndim == 2:
        plt.imshow(img.T, **kwargs)
        plt.show()

    elif ndim == 3:
        img, voxsize = reoirent_numpy_image(img, voxsize, coords)
        midaxial = img.shape[2] // 2
        midcoronal = img.shape[1] // 2
        midsagittal = img.shape[0] // 2
        axial_aspect_ratio = voxsize[1] / voxsize[0]
        coronal_aspect_ratio = voxsize[2] / voxsize[0]
        sagittal_aspect_ratio = voxsize[2] / voxsize[1]

        axial = plt.subplot(2, 3, 1)
        plt.imshow(img[:, :, midaxial].T, **kwargs)
        axial.set_aspect(axial_aspect_ratio)

        coronal = plt.subplot(2, 3, 2)
        plt.imshow(img[:, midcoronal, :].T, **kwargs)
        coronal.set_aspect(coronal_aspect_ratio)

        sagittal = plt.subplot(2, 3, 3)
        plt.imshow(img[midsagittal, :, :].T, **kwargs)
        sagittal.set_aspect(sagittal_aspect_ratio)
        plt.show()

    elif ndim in (4, 5):
        for i in range(img.shape[0]):
            imshow(img[i, ...])


# .....................................................................................................................

def reoirent_numpy_image(img, voxsize=(1, 1, 1), coords=('L', 'A', 'S')):
    """
    This function re-orients a 3D ndarray volume into the standard radiology coordinate system: ('L','A','S').
    """
    assert img.ndim == 3, 'image should have shape (x, y, z)'
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
    raise Exception('coords not identified: please revise the imshow function and define the coordinate system')

# .....................................................................................................................

def one_hot_encode(seg, labels_list):
    """
    This function takes in multi-label segmentation volume and returns one-hot-encoded segmentation volume.
    :param seg: {ndarray} segmentation volume. Shape: (x, y, z)
    :param labels_list: {tuple or list} list of numerical labels.
        e.g. [0, 13, 22, 34, 67]. This list will be converted to one-hot-encoding with the shape (cl, x, y, z),
        with the first label channel lb=0 assigned to the pixels with value 0, the second channel lb=1 assigned to
        the pixels with value 13, etc.
    :return: one-hot-encoded segmentation volume. Shape: (nlb, x, y, z), nlb: number of segmentation labels.
    """
    nlb = len(labels_list)                  # number of labels
    x, y, z = seg.shape
    seg_ohe = np.zeros((nlb, x, y, z))       # segmentation one-hot-encoded pre-allocation
    for i, lb in enumerate(labels_list):
        seg_lb = np.zeros_like(seg)          # label channel pre-allocation
        seg_lb[seg == lb] = 1
        seg_ohe[i, ...] = seg_lb
    return seg_ohe

# .....................................................................................................................

def crop_random_patch(img, patch_shape, return_slicer=False):
    """
    This function returns a random patch with patch_shape cropped from image.
    :param img: {ndarray} image or segmentation
    :param patch_shape: {tuple, list, or ndarray}
    :param return_patching_slicer: {bool}
    :return: patch of the image with patch_shape ± patch slicer
    """
    img_shape, patch_shape =np.array(img.shape), np.array(patch_shape)
    diff = img_shape - patch_shape
    patch_origin = np.random.randint(low=0, high=diff + 1)
    patch_slicer = []
    for dim in range(len(patch_origin)):
        patch_slicer.append(slice(patch_origin[dim], patch_origin[dim] + patch_shape[dim]))
    patch_slicer = tuple(patch_slicer)
    if not return_slicer:
        return img[patch_slicer]
    else:
        return img[patch_slicer], patch_slicer



# .....................................................................................................................

def patch_and_segment(model, image, classes, patch_shape, patch_gaussian_filter, step_size=0.5,
                      pad_border_mode='constant', pad_kwargs=None):
    """
    This function patches and segments an image.
    :param model: segmentation model
    :param image: {tensor} input image to be segmented. Shape: (c, x, y, z)
    :param classes: {tuple, list, ndarray, or tensor} mapping of true multi-labeled classes.
        e.g. [0, 13, 21, 32, 33, 67] maps 0 to 0, 1 to 13, 2 to 21, etc.
        In one-hot-encoded image, the index of the one-hot channel will be the indexes of this list,
        and the values in this list will be the class labels of that one-hot channel.
    :param patch_shape: {tuple, list, ndarray, or tensor}. e.g. (px, py, pz) = (64, 64, 64)
    :param patch_gaussian_filter: {tensor} Shape: (px, py, pz) where p stands for patch.
    :param step_size: {float} step size of sliding patch as a proportion to patch size. Should be between 0 and 1.
    :param pad_border_mode: {string} refer to scipy.ndimage.gaussian_filter --> look up 'mode' of padding.
    :param pad_kwargs: {dict}: to be passed to scipy.ndimage.gaussian_filter
    :return: segmented image. Shape: (x, y, z). Pixel values will contain true class labels.
    """

    # Assertions:
    assert torch.is_tensor(image), 'image type should be torch.Tensor'
    assert torch.is_tensor(patch_gaussian_filter), 'patch_gaussian_filter type should be torch.Tensor'
    assert image.ndim == 4, 'input should have the shape (c, x, y, z)'
    assert len(patch_shape) == 3, 'patch_shape must be in the form (x, y, z)'
    assert patch_shape == patch_gaussian_filter.shape, 'patch_gaussian_filter.shape should be equal to patch_shape'
    assert 0 < step_size <=1, 'step_size should be between 0 and 1'

    if (np.array(patch_shape) > np.array(image.shape)).any():
        image, slicer = pad_nd_image(image, patch_shape, pad_border_mode, pad_kwargs, return_slicer=True)
    else:
        slicer = None

    _, x, y, z = image.shape
    patches_origins = compute_sliding_patches_origins((x, y, z), patch_shape, step_size)

    with torch.no_grad:
        nohc = len(classes)      # ohc: number of one-hot channels
        aggregated_segmentation_probabilities = torch.zeros((nohc, x, y, z))
        '''
        Here we add a first dimension to patch_gaussian_filter since its shape is (x,y,z) and since weighting is 
        the same for all one-hot channels:
        '''
        patch_weighting_map = patch_gaussian_filter[None, ...] if patch_gaussian_filter is not None \
            else torch.ones((1, x, y, z))
        aggregated_weights = torch.zeros((1, x, y, z))

        for x1 in patches_origins[0]:
            x2 = x1 + patch_shape[0]
            for y1 in patches_origins[1]:
                y2 = y1 + patch_shape[1]
                for z1 in patches_origins[2]:
                    z2 = z1 + patch_shape[2]
                    patch = image[:, x1:x2, y1:y2:, z1:z2]
                    patch_segmentation_probabilities = predict_segmentation_probabilities(model, patch)
                    aggregated_segmentation_probabilities[:, x1:x2, y1:y2:, z1:z2] += \
                        patch_weighting_map * patch_segmentation_probabilities
                    aggregated_weights[x1:x2, y1:y2:, z1:z2] += patch_weighting_map

        image_segmentation_probabilities = aggregated_segmentation_probabilities / aggregated_weights

        image_segmentations = torch.zeros((x, y, z))
        for i, c in enumerate(classes):
            image_segmentations[image_segmentation_probabilities[i,...] > 0.5] = c

        if slicer is not None:
            image_segmentations = image_segmentations[slicer]

    return image_segmentations

# .....................................................................................................................

def compute_sliding_patches_origins(image_shape, patch_shape, step_size=0.5):
    """
    :param image_shape: {tuple, list, or ndarray} e.g. (200, 300, 400)
    :param patch_shape: {tuple, list, or ndarray} e.g. (128, 128, 128)
    :param step_size: size of the sliding window as a fraction of patch size. e.g. step_size=0.6 means each new sliding
         patch overlapts with the previous sliding patch 40%. Default = 0.5
    :return: list of the index of the first patch voxels in each dimension.
        example: [[0, 64, 128], [0, 39, 78, 117, 156], [0, 100, 200, 300]]
    """
    patch_shape, image_shape = np.array(patch_shape), np.array(image_shape)
    assert (patch_shape <= image_shape).all(), "image size is smaller than patch size"
    assert 0 < step_size <= 1, 'step_size must be between 0 and 1'
    # our step width is patch_size*step_size at most, but can be narrower. For example if we have image size of
    # 110, patch size of 64 and step_size of 0.5, then we want to make 3 steps starting at coordinate 0, 23, 46
    target_step_sizes_in_voxels = patch_shape * step_size                            # ndarray[x, y, z]
    num_steps = np.ceil((image_shape - patch_shape) / target_step_sizes_in_voxels + 1).astype('uint8')
    sliding_patches_origins = []
    for dim in range(len(patch_shape)):
        # the last sliding patch origin for this dimension is
        last_patch_origin_dim = image_shape[dim] - patch_shape[dim]
        if num_steps[dim] > 1:
            actual_step_size = last_patch_origin_dim / (num_steps[dim] - 1)
            sliding_patches_origins_dim = [int(np.round(actual_step_size * i)) for i in range(num_steps[dim])]
        else:
            sliding_patches_origins_dim = 0

        sliding_patches_origins.append(sliding_patches_origins_dim)

    return sliding_patches_origins

# .....................................................................................................................

def compute_patch_gaussian(patch_shape, sigma_scale=1/8):
    """
    This function returns a multidimensional Gaussian with 1 centered at the center of the patch, and sigma
    determined by sigma_scale as a proportion of the patch size in each dimension.
    :param patch_shape: {tuple, list, or ndarray}. e.g. (200, 300, 400)
    :param sigma_scale: sigma for the Gaussian
    :return: gaussian filter with 1 centered at the center of the patch. Shape exactly same as patch_size. Beware that
        the integral of this gaussian filter is not 1, this is rather an importance map.
    """
    patch_shape = np.array(patch_shape)
    tmp = np.zeros(patch_shape)
    center_coords = patch_shape // 2
    sigmas = patch_shape * sigma_scale
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, mode='constant', cval=0)
    gaussian_importance_map = (gaussian_importance_map / np.max(gaussian_importance_map)).astype('float32')
    # gaussian_importance_map cannot be 0, otherwise we may end up with nans when we divide by 0
    gaussian_importance_map[gaussian_importance_map == 0] = \
        np.min(gaussian_importance_map[gaussian_importance_map != 0])
    return gaussian_importance_map

# .....................................................................................................................

def pad_nd_image(image, new_shape=None, mode="constant", kwargs=None, return_slicer=False,
                 shape_must_be_divisible_by=None):
    """
    :param image: {np.ndarray or torch.Tensor} image or segmentation to be padded.
    :param new_shape: shape of padded image. If len(new_shape) < len(image.shape) then the last axes of image
    will be padded. If new_shape < image.shape in any of the axes then we will not pad that axis, but also not crop!
    (interpret new_shape as new_min_shape)
    Example:
    image.shape = (10, 1, 512, 512); new_shape = (768, 768) -> result: (10, 1, 768, 768). Cool, huh?
    image.shape = (10, 1, 512, 512); new_shape = (364, 768) -> result: (10, 1, 512, 768).
    :param mode: see np.pad for documentation
    :param return_slicer: if True then this function will also return what coords you will need to use when cropping
    back to original shape
    :param shape_must_be_divisible_by: for network prediction. After applying new_shape, make sure the new shape is
    divisibly by that number (can also be a list with an entry for each axis). Whatever is missing to match that will
    be padded (so the result may be larger than new_shape if shape_must_be_divisible_by is not None)
    :param kwargs: see np.pad for documentation
    :return padded_image: padded image with new_shape
    :return slicer: {tuple, returned if return_slicer=True} the slicers in each x, y, and z dimensions that
    slice the padded image back to original image. e.g. for image.shape=[100, 200, 300] and
    new_shape=[150, 250, 350], the slicer would be: (slice(25, 125, None), slice(25, 225, None), slice(25, 325, None)).
    """
    if kwargs is None:
        kwargs = dict(constant_values=0)

    if new_shape is not None:
        old_shape = np.array(image.shape[-len(new_shape):])
    else:
        assert shape_must_be_divisible_by is not None
        assert isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray))
        new_shape = image.shape[-len(shape_must_be_divisible_by):]
        old_shape = new_shape

    num_axes_nopad = image.ndim - len(new_shape)

    new_shape = [max(new_shape[i], old_shape[i]) for i in range(len(new_shape))]

    new_shape = np.array(new_shape)

    if shape_must_be_divisible_by is not None:
        if not isinstance(shape_must_be_divisible_by, (list, tuple, np.ndarray)):
            shape_must_be_divisible_by = [shape_must_be_divisible_by] * len(new_shape)
        else:
            assert len(shape_must_be_divisible_by) == len(new_shape)

        for i in range(len(new_shape)):
            if new_shape[i] % shape_must_be_divisible_by[i] == 0:
                new_shape[i] -= shape_must_be_divisible_by[i]

        new_shape = np.array([new_shape[i] + shape_must_be_divisible_by[i] - new_shape[i]
                              % shape_must_be_divisible_by[i] for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference // 2 + difference % 2
    pad_list = [[0, 0]]*num_axes_nopad + list([list(i) for i in zip(pad_below, pad_above)])

    if not ((all([i == 0 for i in pad_below])) and (all([i == 0 for i in pad_above]))):
        res = np.pad(image, pad_list, mode, **kwargs)
    else:
        res = image

    if not return_slicer:
        return res
    else:
        pad_list = np.array(pad_list)
        pad_list[:, 1] = np.array(res.shape) - pad_list[:, 1]
        slicer = tuple(slice(*i) for i in pad_list)
        return res, slicer



# -------------------------------------------------- CODE TESTING -----------------------------------------------------

if __name__ == '__main__':

    image_shape = (20, 30, 80)
    patch_shape = (20, 30, 40)
    sliding_patches_origins = compute_sliding_patches_origins(image_shape, patch_shape)
    patch_gauss = compute_patch_gaussian(patch_shape)

    print(f'''
    sliding patches origins: {sliding_patches_origins}
    patch_gaussian shape: {patch_gauss.shape}
    patch_gaussian max: {patch_gauss.max()}     location: {np.where(patch_gauss==1)}
    ''')

    image = np.zeros((100, 200, 300))
    image2 = np.array(image)
    res, slicer = pad_nd_image(image, new_shape=(150, 250, 350), return_slicer=True)
    print(f'''
    padded_image shape:    {res.shape}
    slicer: {slicer}
    ''')
