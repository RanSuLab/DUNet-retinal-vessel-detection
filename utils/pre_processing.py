###################################################
#
#   Script to pre-process the original imgs
#
##################################################
import cv2
from utils.help_functions import *
from os import path
from skimage import io, color, measure
from scipy import ndimage, stats


# My pre processing (use for both training and testing!)
def my_PreProc(data):
    assert (len(data.shape) == 4)
    assert (data.shape[1] == 3)  # Use the original images
    train_imgs = rgb2gray(data)
    # my preprocessing:
    train_imgs = dataset_normalized(train_imgs)
    train_imgs = clahe_equalized(train_imgs)
    train_imgs = adjust_gamma(train_imgs, 1.2)
    train_imgs = train_imgs / 255.  # reduce to 0-1 range
    return train_imgs


# ============================================================
# ========= PRE PROCESSING FUNCTIONS ========================#
# ============================================================

# ==== histogram equalization
def histo_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    assert (imgs.shape[1] == 1)  # check the channel is 1
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = cv2.equalizeHist(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization is used. In this, image is divided into small blocks called "tiles" (tileSize is 8x8 by default in OpenCV). Then each of these blocks are histogram equalized as usual. So in a small area, histogram would confine to a small region (unless there is noise). If noise is there, it will be amplified. To avoid this, contrast limiting is applied. If any histogram bin is above the specified contrast limit (by default 40 in OpenCV), those pixels are clipped and distributed uniformly to other bins before applying histogram equalization. After equalization, to remove artifacts in tile borders, bilinear interpolation is applied
def clahe_equalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    imgs_equalized = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        imgs_equalized[i, 0] = clahe.apply(np.array(imgs[i, 0], dtype=np.uint8))
    return imgs_equalized


# ===== normalize over the dataset
def dataset_normalized(imgs):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    imgs_normalized = np.empty(imgs.shape)
    imgs_std = np.std(imgs)
    imgs_mean = np.mean(imgs)
    imgs_normalized = (imgs - imgs_mean) / imgs_std
    for i in range(imgs.shape[0]):
        imgs_normalized[i] = ((imgs_normalized[i] - np.min(imgs_normalized[i])) / (
                    np.max(imgs_normalized[i]) - np.min(imgs_normalized[i]))) * 255
    return imgs_normalized


def adjust_gamma(imgs, gamma=1.0):
    assert (len(imgs.shape) == 4)  # 4D arrays
    # assert (imgs.shape[1]==1)  #check the channel is 1
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    new_imgs = np.empty(imgs.shape)
    for i in range(imgs.shape[0]):
        new_imgs[i, 0] = cv2.LUT(np.array(imgs[i, 0], dtype=np.uint8), table)
    return new_imgs


def get_fov_mask(image_rgb, threshold=0.01):
    '''
    Automatically calculate the FOV mask (see Orlando et al., SIPAIM 2016 for further details) Convolutional neural network transfer for automated glaucoma identification
    '''
    # format: [H, W, #channels]
    image_lab = color.rgb2lab(image_rgb)
    # normalize the luminosity plane
    image_lab[:, :, 0] /= 100.0
    # threshold the plane at the given threshold
    mask = image_lab[:, :, 0] >= threshold

    # fill holes in the resulting mask
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.filters.median_filter(mask, size=(5, 5))

    # get connected components
    connected_components = measure.label(mask).astype(float)

    # replace background found in [0][0] to nan so mode skips it
    connected_components[connected_components == mask[0][0]] = np.nan

    # get largest connected component (== mode of the image)
    largest_component_label = stats.mode(connected_components, axis=None, nan_policy='omit')[0]

    # use the modal value of the labels as the final mask
    mask = connected_components == largest_component_label

    return mask.astype(float)


def generate_fov_masks(image_path, image_filenames, threshold=0.01):
    '''
    Generate FOV masks for all the images in image_filenames
    '''

    for i in range(0, len(image_filenames)):
        # get current filename
        current_filename = path.basename(image_filenames[i])
        # read the image
        img = io.imread(path.join(image_path, current_filename))
        # get fov mask
        fov_mask = get_fov_mask(img, threshold)
        # save the fov mask
        io.imsave(path.join(image_path, current_filename[:-4] + '_fov_mask.png'), fov_mask)
