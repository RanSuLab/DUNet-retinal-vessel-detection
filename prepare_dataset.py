import os
import h5py
import numpy as np
from pre_processing import get_fov_mask

np.random.seed(1337)
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
dataset = config.get('data attributes', 'dataset')
dataset_dict = ['STARE', 'CHASE']


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# train
original_imgs_train = "./" + dataset + "/training/images/"
groundTruth_imgs_train = "./" + dataset + "/training/1st_manual/"
borderMasks_imgs_train = "./" + dataset + "/training/mask/"
# test
original_imgs_test = "./" + dataset + "/test/images/"
groundTruth_imgs_test = "./" + dataset + "/test/1st_manual/"
borderMasks_imgs_test = "./" + dataset + "/test/mask/"

Nimgs = 0
channels = int(config.get('data attributes', 'channels'))
height = 0
width = 0

dataset_path = "./" + dataset + "_datasets_training_testing/"


def get_datasets(imgs_dir, groundTruth_dir, borderMasks_dir, train_test="null"):
    # for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
    files = os.listdir(imgs_dir)
    assert len(files) > 0
    img = Image.open(imgs_dir + files[0])
    sp = np.asarray(img).shape
    Nimgs = len(files)
    height = sp[0]
    width = sp[1]
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))
    border_masks = np.empty((Nimgs, height, width))
    for i in range(len(files)):
        # original
        print("original image: " + files[i])
        img = Image.open(imgs_dir + files[i])
        imgs[i] = np.asarray(img)
        # corresponding ground truth
        if dataset == "STARE":
            groundTruth_name = files[i]
        if dataset == "DRIVE":
            groundTruth_name = files[i][0:2] + "_manual1.gif"
        if dataset == "CHASE":
            groundTruth_name = files[i][0:len(files[i]) - 4] + ".png"
        if dataset == "HRF":
            groundTruth_name = files[i][:-4] + ".tif"
        if dataset == "SYNTHE":
            groundTruth_name = files[i][0:2] + "_manual1.gif"

        print("ground truth name: " + groundTruth_name)

        g_truth = Image.open(groundTruth_dir + groundTruth_name)
        groundTruth[i] = np.asarray(g_truth)

        # corresponding border masks for DRIVE HRF SYNTHE
        if dataset not in dataset_dict:
            border_masks_name = ""
            if dataset == "DRIVE" or dataset == "SYNTHE":
                if train_test == "train":
                    border_masks_name = files[i][0:2] + "_training_mask.gif"
                elif train_test == "test":
                    border_masks_name = files[i][0:2] + "_test_mask.gif"
                else:
                    print("specify if train or test!!")
                    exit()
            if dataset == "HRF":
                border_masks_name = files[i][:-4] + "_mask.tif"
            print("border masks name: " + border_masks_name)
            b_mask = Image.open(borderMasks_dir + border_masks_name)
            if np.asarray(b_mask).shape[-1] == 3:
                b_mask = np.asarray(b_mask)[..., 0]
            border_masks[i] = np.asarray(b_mask)
        else:
            # get fov mask for STARE CHASE
            threshold = 0.01
            if dataset == "STARE":
                threshold = 0.19
                # threshold = 0.19
            fov_mask = get_fov_mask(img, threshold=threshold)
            border_masks[i] = np.asarray(fov_mask)
            # save the fov mask
            Image.fromarray(fov_mask * 255).convert("RGB").save(
                borderMasks_imgs_test + files[i][:-4] + '_fov_mask.png', "png")

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    if np.max(groundTruth) == 1.0:
        groundTruth = groundTruth * 255
    assert (int(np.max(groundTruth)) == 255)
    assert (int(np.min(groundTruth)) == 0)
    print("ground truth are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
    border_masks = np.reshape(border_masks, (Nimgs, 1, height, width))
    assert (groundTruth.shape == (Nimgs, 1, height, width))
    assert (border_masks.shape == (Nimgs, 1, height, width))
    return imgs, groundTruth, border_masks


if not os.path.exists(dataset_path):
    os.makedirs(dataset_path)
# imgs_train, groundTruth_train, border_masks_train = get_datasets(original_imgs_train, groundTruth_imgs_train, borderMasks_imgs_train)
# write_hdf5(imgs_train, dataset_path + dataset + "_imgs_train.hdf5")
# write_hdf5(groundTruth_train, dataset_path + dataset + "_groundTruth_train.hdf5")
# write_hdf5(border_masks_train, dataset_path + dataset + "_borderMasks_train.hdf5")

imgs_test, groundTruth_test, border_masks_test = get_datasets(original_imgs_test, groundTruth_imgs_test,
                                                              borderMasks_imgs_test, "test")
write_hdf5(imgs_test, dataset_path + dataset + "_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + dataset + "_groundTruth_test.hdf5")
write_hdf5(border_masks_test, dataset_path + dataset + "_borderMasks_test.hdf5")
