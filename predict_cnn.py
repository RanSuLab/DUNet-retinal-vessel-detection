import numpy as np
import configparser
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from build_model import build_deform_cnn

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('tf')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
import sys

sys.path.insert(0, './utils/')
from help_functions import *
from extract_patches import get_data_testing_single_image

# ========= CONFIG FILE TO READ FROM =======
config = configparser.ConfigParser()
config.read('configuration.txt')
algorithm = 'deform'
dataset = config.get('data attributes', 'dataset')
path_experiment = './log/experiments/' + algorithm + '/'+dataset+'/'
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
test_imgs_orig = load_hdf5(test_imgs_original)

gtruth = path_data + config.get('data paths', 'test_groundTruth')
img_truth = load_hdf5(gtruth)

full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
print(test_imgs_orig.shape, full_img_width, full_img_height)

# # the border masks provided by the DRIVE
# DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
# test_border_masks = load_hdf5(DRIVE_test_border_masks)

# dimension of the patches
patch_height = 29
patch_width = 29

# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

print(N_visual, path_experiment)

visualize(group_images(test_imgs_orig[0:14, :, :, :], 7), path_experiment + 'test_original')

visualize(group_images(img_truth[0:14, :, :, :], 7), path_experiment + 'test_gtruth')

# Load the saved model
model = build_deform_cnn((patch_height, patch_width, 1))
model.load_weights(path_experiment + 'model_best.hdf5')
pred_imgs = np.zeros(img_truth.shape)
for index in range(test_imgs_orig.shape[0]):
    # for index in range(1):
    print(index + 1, " image")
    # ============ Load the data and divide in patches
    patches_imgs_test = get_data_testing_single_image(
        test_imgs_original=test_imgs_original,  # original
        test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
        patch_height=patch_height,
        patch_width=patch_width,
        index=index
    )

    print(patches_imgs_test.shape)

    # ================ Run the prediction of the patches ==================================

    # Calculate the predictions
    predictions = model.predict(patches_imgs_test, batch_size=1920, verbose=1)
    print("predicted images size :")
    print(predictions.shape)

    # ===== Convert the prediction arrays in corresponding images
    pred_img = conv_to_imgs(pred=predictions, img_h=img_truth.shape[2], img_w=img_truth.shape[3], mode='original',
                            patch_h=patch_height, patch_w=patch_width, path_experiment=path_experiment, index=index)
    pred_imgs[index, :, :, :] = pred_img

file = h5py.File(path_experiment +dataset+'_predict_results.h5', 'w')
file.create_dataset('y_gt', data=img_truth)
file.create_dataset('y_pred', data=pred_imgs)
file.create_dataset('x_origin', data=test_imgs_orig)
file.close()
