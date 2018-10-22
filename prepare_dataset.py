import os
import h5py
import numpy as np

np.random.seed(1337)
from PIL import Image
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
dataset = config.get('data attributes', 'dataset')


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# train
original_imgs_train = "./" + dataset + "/training/images/"
groundTruth_imgs_train = "./" + dataset + "/training/1st_manual/"

# test
original_imgs_test = "./" + dataset + "/test/images/"
groundTruth_imgs_test = "./" + dataset + "/test/1st_manual/"

Nimgs = int(config.get('data attributes', 'Nimgs'))
channels = int(config.get('data attributes', 'channels'))
height = int(config.get('data attributes', 'height'))
width = int(config.get('data attributes', 'width'))

dataset_path = "./" + dataset + "_datasets_training_testing/"


def get_datasets(imgs_dir, groundTruth_dir):
    imgs = np.empty((Nimgs, height, width, channels))
    groundTruth = np.empty((Nimgs, height, width))

    for path, subdirs, files in os.walk(imgs_dir):  # list all files, directories in the path
        for i in range(len(files)):
            # original
            print("original image: " + files[i])
            img = Image.open(imgs_dir + files[i])
            imgs[i] = np.asarray(img)
            # corresponding ground truth
            # groundTruth_name = files[i]
            groundTruth_name = files[i][0:2] + "_manual1.gif"
            # groundTruth_name = files[i][0:len(files[i]) - 4] + ".png"
            print("ground truth name: " + groundTruth_name)

            g_truth = Image.open(groundTruth_dir + groundTruth_name)
            groundTruth[i] = np.asarray(g_truth)

    print("imgs max: " + str(np.max(imgs)))
    print("imgs min: " + str(np.min(imgs)))
    if np.max(groundTruth)==1.0:
        groundTruth=groundTruth*255
    assert (int(np.max(groundTruth)) == 255)
    assert (int(np.min(groundTruth)) == 0)
    print("ground truth are correctly withih pixel value range 0-255 (black-white)")
    # reshaping for my standard tensors
    imgs = np.transpose(imgs, (0, 3, 1, 2))
    assert (imgs.shape == (Nimgs, channels, height, width))
    groundTruth = np.reshape(groundTruth, (Nimgs, 1, height, width))
    assert (groundTruth.shape == (Nimgs, 1, height, width))
    return imgs, groundTruth


# imgs_train, groundTruth_train = get_datasets(original_imgs_train, groundTruth_imgs_train)
# write_hdf5(imgs_train, dataset_path + dataset + "_imgs_train.hdf5")
# write_hdf5(groundTruth_train, dataset_path + dataset + "_groundTruth_train.hdf5")

imgs_test, groundTruth_test = get_datasets(original_imgs_test, groundTruth_imgs_test)
write_hdf5(imgs_test, dataset_path + dataset + "_imgs_test.hdf5")
write_hdf5(groundTruth_test, dataset_path + dataset + "_groundTruth_test.hdf5")
