# Python
import sys
from os.path import isdir, join
from os import makedirs
from utils.Data_loader import Retina_loader_infer
from torch.utils.data import DataLoader
import os

sys.path.insert(0, './utils/')
from models import MODELS
# help_functions.py
from help_functions import *
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import kill_border
from extract_patches import get_data_testing, get_data_testing_overlap
from pre_processing import my_PreProc
import time
from glob import glob
import torch
# ========= CONFIG FILE TO READ FROM =======
import configparser
import argparse

parser = argparse.ArgumentParser(description="nasopharyngeal training")
parser.add_argument('--mode', default='gpu', type=str, metavar='train on gpu or cpu',
                    help='train on gpu or cpu(default gpu)')
parser.add_argument('--gpu', default=0, type=int, help='gpu number')
args = parser.parse_args()

gpuid = args.gpu
mode = args.mode

config = configparser.ConfigParser()
config.read('configuration.txt')
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

# original test images (for FOV selection)
test_imgs_original = path_data + config.get('data paths', 'test_imgs_original')
print("Test data:" + test_imgs_original)
test_imgs_orig = load_hdf5(test_imgs_original)
full_img_height = test_imgs_orig.shape[2]
full_img_width = test_imgs_orig.shape[3]
# the border masks provided by the DRIVE
test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(test_border_masks)
# dimension of the patches
patch_height = int(config.get('data attributes', 'patch_height'))
patch_width = int(config.get('data attributes', 'patch_width'))
# the stride in case output with average
stride_height = int(config.get('testing settings', 'stride_height'))
stride_width = int(config.get('testing settings', 'stride_width'))
assert (stride_height < patch_height and stride_width < patch_width)
# model name
name_experiment = config.get('experiment name', 'name')
dataset = config.get('data attributes', 'dataset')
path_experiment = './log/experiments/' + name_experiment + '/' + dataset + '/'
# Grouping of the predicted images
N_visual = int(config.get('testing settings', 'N_group_visual'))
# ====== average mode ===========
average_mode = config.getboolean('testing settings', 'average_mode')

TMP_DIR = path_experiment
if not isdir(TMP_DIR):
    makedirs(TMP_DIR)


def to_cuda(t, mode):
    if mode == 'gpu':
        return t.cuda()
    return t


# #ground truth
# gtruth= path_data + config.get('data paths', 'test_groundTruth')
# img_truth= load_hdf5(gtruth)
# visualize(group_images(test_imgs_orig[0:20,:,:,:],5),'original')#.show()
# visualize(group_images(test_border_masks[0:20,:,:,:],5),'borders')#.show()
# visualize(group_images(img_truth[0:20,:,:,:],5),'gtruth')#.show()


# ============ Load the data and divide in patches
patches_imgs_test = None
new_height = None
new_width = None
masks_test = None
patches_masks_test = None
if average_mode == True:
    patches_imgs_test, new_height, new_width, masks_test = get_data_testing_overlap(
        test_imgs_original=test_imgs_original,  # original
        test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
        patch_height=patch_height,
        patch_width=patch_width,
        stride_height=stride_height,
        stride_width=stride_width
    )
else:
    patches_imgs_test, patches_masks_test = get_data_testing(
        test_imgs_original=test_imgs_original,  # original
        test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
        patch_height=patch_height,
        patch_width=patch_width,
    )

# ================ Run the prediction of the patches ==================================
batch_size = int(config.get('training settings', 'batch_size'))

model = MODELS[name_experiment](n_channels=1, n_classes=1)

weight_files = sorted(glob(join(TMP_DIR, 'checkpoint_epoch_*.pth')), reverse=True)
# weight_files = []
# weight_files.append(join(TMP_DIR, 'checkpoint_epoch_006.pth'))
print("loaded:" + weight_files[0])
if mode == 'cpu':
    model.load_state_dict(torch.load(weight_files[0],
                                     map_location={'cuda:0': 'cpu', 'cuda:1': 'cpu',
                                                   'cuda:2': 'cpu', 'cuda:3': 'cpu'})['state_dict'])
    dtype_float = torch.FloatTensor

else:
    torch.cuda.set_device(gpuid)
    model.load_state_dict(torch.load(weight_files[0], map_location=('cuda:' + str(gpuid)))['state_dict'])
    model.cuda()
    dtype_float = torch.cuda.FloatTensor
model.eval()
# Load the saved model
# Calculate the predictions
test_dataset = Retina_loader_infer(patches_imgs_test)
test_loader = DataLoader(test_dataset, batch_size=batch_size * 1, shuffle=False)
# Calculate the predictions
start_time = time.time()
predictions = []
with torch.no_grad():
    for i, (image) in enumerate(test_loader):
        image = dtype_float(to_cuda(image.float(), mode)).requires_grad_(False)
        pre_label = model(image)
        pred_prob = pre_label.cpu().detach().numpy()
        # visualize(group_images(image.cpu().detach().numpy(), 20), path_experiment + "all_testtest_" + str(i)+"_A")  # .show()
        # visualize(group_images(pred_prob, 20),
        #           path_experiment + "all_testtest_" + str(i)+"_B")  # .show()
        # _, preds = torch.max(pre_label, dim=1)
        # predictions.append(pred_prob.cpu().numpy())
        predictions.append(pred_prob)
end_time = time.time()
print("predict time:" + str(end_time - start_time))
# ===== Convert the prediction arrays in corresponding images
print("predicted images size :")
pred_patches = np.concatenate(predictions, 0)
print(pred_patches.shape)

# ========== Elaborate and visualize the predicted images ====================
pred_imgs = None
orig_imgs = None
gtruth_masks = None
if average_mode == True:
    pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height, stride_width)  # predictions
    orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals
    gtruth_masks = np.transpose(masks_test, (0, 3, 1, 2))  # ground truth masks

else:
    pred_imgs = recompone(pred_patches, 13, 12)  # predictions
    orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
    gtruth_masks = recompone(np.transpose(patches_masks_test, (0, 3, 1, 2)), 13, 12)  # masks
# apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
# kill_border(pred_imgs, test_border_masks)  # MASK  #only for visualization
## back to original dimensions


orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
gtruth_masks = gtruth_masks[:, :, 0:full_img_height, 0:full_img_width]

print("Orig imgs shape: " + str(orig_imgs.shape))
print("pred imgs shape: " + str(pred_imgs.shape))
print("Gtruth imgs shape: " + str(gtruth_masks.shape))
# visualize(group_images(test_imgs_orig, N_visual), path_experiment + "all_originals_RGB")  # .show()
# visualize(group_images(orig_imgs, N_visual), path_experiment + "all_originals")  # .show()
# visualize(group_images(pred_imgs, N_visual), path_experiment + "all_predictions")  # .show()
# visualize(group_images(gtruth_masks, N_visual), path_experiment + "all_groundTruths")  # .show()
# visualize results comparing mask and prediction:
assert (orig_imgs.shape[0] == pred_imgs.shape[0] and orig_imgs.shape[0] == gtruth_masks.shape[0])
N_predicted = orig_imgs.shape[0]
group = N_visual
assert (N_predicted % group == 0)
for i in range(int(N_predicted / group)):
    orig_rgb_stripe = group_images(test_imgs_orig[i * group:(i * group) + group, :, :, :], group) / 255.
    orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
    masks_stripe = group_images(gtruth_masks[i * group:(i * group) + group, :, :, :], group)
    pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
    total_img = np.concatenate(
        (orig_rgb_stripe, np.tile(orig_stripe, 3), np.tile(masks_stripe, 3), np.tile(pred_stripe, 3)), axis=0)
    visualize(total_img, path_experiment + name_experiment + "_RGB_Original_GroundTruth_Prediction" + str(i))  # .show()

file = h5py.File(path_experiment + dataset + '_predict_results.h5', 'w')
file.create_dataset('y_gt', data=gtruth_masks)
file.create_dataset('y_pred', data=pred_imgs)
file.create_dataset('x_origin', data=test_imgs_orig)
file.close()
