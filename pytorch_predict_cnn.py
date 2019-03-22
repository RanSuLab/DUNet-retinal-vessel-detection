# Python
import sys, os
from os.path import isdir, join
from os import makedirs
from torch.cuda import empty_cache

sys.path.insert(0, './utils/')
from models import MODELS
import torch.nn.functional as F
# help_functions.py
import time
from glob import glob
import torch
from torch.utils.data import DataLoader
from Data_loader import Retina_loader_infer
# ========= CONFIG FILE TO READ FROM =======
import configparser
import sys
import argparse

sys.path.insert(0, './utils/')
from help_functions import *
from extract_patches import get_data_testing_single_image

parser = argparse.ArgumentParser(description="nasopharyngeal training")
parser.add_argument('--mode', default='gpu', type=str, metavar='train on gpu or cpu',
                    help='train on gpu or cpu(default gpu)')
parser.add_argument('--gpu', default=1, type=int, help='gpu number')
args = parser.parse_args()

gpuid = args.gpu
mode = args.mode

# ========= CONFIG FILE TO READ FROM =======
config = configparser.ConfigParser()
config.read('configuration.txt')
algorithm = config.get('experiment name', 'name')
dataset = config.get('data attributes', 'dataset')
path_experiment = './log/experiments/' + algorithm + '/' + dataset + '/'
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
print("Test data:" + test_imgs_original)
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
TMP_DIR = path_experiment
if not isdir(TMP_DIR):
    makedirs(TMP_DIR)


def to_cuda(t, mode):
    if mode == 'gpu':
        return t.cuda()
    return t


# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))

print(N_visual, path_experiment)

# Load the saved model

model = MODELS[algorithm](n_channels=1, n_classes=2)

weight_files = sorted(glob(join(TMP_DIR, 'checkpoint_epoch_*.pth')), reverse=True)
# weight_files = []
# weight_files.append(join(TMP_DIR, 'checkpoint_epoch_013.pth'))
print("loaded:"+weight_files[0])
if mode == 'cpu':
    dtype_float = torch.FloatTensor
else:
    torch.cuda.set_device(gpuid)
    model.load_state_dict(torch.load(weight_files[0], map_location=('cuda:' + str(gpuid)))['state_dict'])
    model.cuda()
    dtype_float = torch.cuda.FloatTensor
model.eval()
pred_imgs = np.zeros(img_truth.shape)
test_imgs_original_hdf5 = load_hdf5(test_imgs_original)
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
    test_dataset = Retina_loader_infer(patches_imgs_test)
    test_loader = DataLoader(test_dataset, batch_size=200, shuffle=False)
    # Calculate the predictions
    start_time = time.time()
    predictions = []
    with torch.no_grad():
        for i, (image) in enumerate(test_loader):
            image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
            pre_label = model(image)
            pred_prob = F.softmax(pre_label, dim=1).cpu().detach().numpy()
            # _, preds = torch.max(pre_label, dim=1)
            # predictions.append(pred_prob.cpu().numpy())
            predictions.append(pred_prob)
    end_time = time.time()
    print("predict time:" + str(end_time - start_time))
    # for patch_p in patches_imgs_test:
    #     patches_imgs_test = dtype_float(
    #         to_cuda(torch.from_numpy(np.expand_dims(patch_p, 0)).float(), mode)).requires_grad_(
    #         False)
    #     predictions.append(model(patches_imgs_test))
    predictions = np.concatenate(predictions, 0)
    print("predicted images size :")
    print(predictions.shape)
    empty_cache()

    # ===== Convert the prediction arrays in corresponding images
    pred_img = conv_to_imgs(pred=predictions, img_h=img_truth.shape[2], img_w=img_truth.shape[3], mode='original',
                            patch_h=patch_height, patch_w=patch_width, path_experiment=path_experiment, index=index)
    pred_imgs[index, :, :, :] = pred_img

file = h5py.File(path_experiment + dataset + '_predict_results.h5', 'w')
file.create_dataset('y_gt', data=img_truth)
file.create_dataset('y_pred', data=pred_imgs)
file.create_dataset('x_origin', data=test_imgs_orig)
file.close()
