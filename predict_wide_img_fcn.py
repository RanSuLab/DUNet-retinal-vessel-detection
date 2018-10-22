# Python
import sys

sys.path.insert(0, './utils/')
from keras.models import model_from_json
# help_functions.py
from help_functions import *
from extract_patches import recompone
from extract_patches import recompone_overlap
from extract_patches import get_data_testing, get_data_testing_overlap
from pre_processing import my_PreProc
import os
import tensorflow as tf
from build_model import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
# ========= CONFIG FILE TO READ FROM =======
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
# ===========================================
# run the training on invariant or local
path_data = config.get('data paths', 'path_local')

dataset = 'WIDE'
original_imgs_train = "./" + dataset + "/test/images/"
dataset_path = "./" + dataset + "_datasets_training_testing/"
for path, subdirs, files in os.walk(original_imgs_train):  # list all files, directories in the path
    for k in range(len(files)):
        # original
        print("original image: " + files[k])
        test_imgs_orig = Image.open(original_imgs_train + files[k])
        test_imgs_orig = np.asarray(test_imgs_orig)
        print("imgs max: " + str(np.max(test_imgs_orig)))
        print("imgs min: " + str(np.min(test_imgs_orig)))
        test_imgs_orig = np.expand_dims(test_imgs_orig, axis=0)
        # reshaping for my standard tensors
        test_imgs_orig = np.transpose(test_imgs_orig, (0, 3, 1, 2))
        write_hdf5(test_imgs_orig, dataset_path + dataset + "_imgs_test.hdf5")
        test_imgs_original = dataset_path + dataset + "_imgs_test.hdf5"

        # original test images (for FOV selection)
        full_img_height = test_imgs_orig.shape[2]
        full_img_width = test_imgs_orig.shape[3]
        # the border masks provided by the DRIVE
        # DRIVE_test_border_masks = path_data + config.get('data paths', 'test_border_masks')
        # test_border_masks = load_hdf5(DRIVE_test_border_masks)
        # dimension of the patches
        patch_height = int(config.get('data attributes', 'patch_height'))
        patch_width = int(config.get('data attributes', 'patch_width'))
        # the stride in case output with average
        stride_height = int(config.get('testing settings', 'stride_height'))
        stride_width = int(config.get('testing settings', 'stride_width'))
        assert (stride_height < patch_height and stride_width < patch_width)
        # model name
        name_experiment = config.get('experiment name', 'name')
        path_experiment = './log/experiments/' + name_experiment + '/' + dataset + '/'
        # N full images to be predicted
        Imgs_to_test = 1
        # Grouping of the predicted images
        N_visual = 1
        # ====== average mode ===========
        average_mode = config.getboolean('testing settings', 'average_mode')
        crf = config.getboolean('testing settings', 'crf')

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
                Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
                patch_height=patch_height,
                patch_width=patch_width,
                stride_height=stride_height,
                stride_width=stride_width
            )
        else:
            patches_imgs_test, patches_masks_test = get_data_testing(
                test_imgs_original=test_imgs_original,  # original
                test_groudTruth=path_data + config.get('data paths', 'test_groundTruth'),  # masks
                Imgs_to_test=int(config.get('testing settings', 'full_images_to_test')),
                patch_height=patch_height,
                patch_width=patch_width,
            )

        # ================ Run the prediction of the patches ==================================
        # Load the saved model
        model = BuildR2UNet((patch_height, patch_width, 1))
        model.load_weights('./log/experiments/'+name_experiment+'/DRIVE/model_best.hdf5')
        # Calculate the predictions
        predictions = model.predict(patches_imgs_test, batch_size=480, verbose=1)
        print("predicted images size :")
        print(predictions.shape)

        # ===== Convert the prediction arrays in corresponding images
        # pred_patches = pred_to_imgs(predictions, patch_height, patch_width, "original")
        pred_patches = np.transpose(predictions, (0, 3, 1, 2))

        # ========== Elaborate and visualize the predicted images ====================
        pred_imgs = None
        orig_imgs = None
        gtruth_masks = None
        if average_mode == True:
            pred_imgs = recompone_overlap(pred_patches, new_height, new_width, stride_height,
                                          stride_width)  # predictions
            orig_imgs = my_PreProc(test_imgs_orig[0:pred_imgs.shape[0], :, :, :])  # originals

        else:
            pred_imgs = recompone(pred_patches, 13, 12)  # predictions
            orig_imgs = recompone(patches_imgs_test, 13, 12)  # originals
        # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
        # kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
        ## back to original dimensions

        orig_imgs = orig_imgs[:, :, 0:full_img_height, 0:full_img_width]
        pred_imgs = pred_imgs[:, :, 0:full_img_height, 0:full_img_width]
        # if crf:
        #     crf_preds = np.zeros(pred_imgs.shape)
        #     for i in range(Imgs_to_test):
        #         crf_pr = dense_crf(np.transpose(test_imgs_orig[i], (1, 2, 0)), pred_imgs[i, 0, :, :])
        #         # crf_pr = dense_crf_no_rgb(np.transpose(orig_imgs[i], (1, 2, 0)), pred_imgs[i, 0, :, :])
        #         crf_preds[i][0] = crf_pr
        #     pred_imgs=crf_preds
        print("Orig imgs shape: " + str(orig_imgs.shape))
        print("pred imgs shape: " + str(pred_imgs.shape))
        visualize(group_images(test_imgs_orig, N_visual), path_experiment + "all_originals_RGB")  # .show()
        visualize(group_images(orig_imgs, N_visual), path_experiment + "all_originals")  # .show()
        visualize(group_images(pred_imgs, N_visual), path_experiment + "all_predictions")  # .show()
        # visualize results comparing mask and prediction:
        N_predicted = orig_imgs.shape[0]
        group = N_visual
        assert (N_predicted % group == 0)
        for i in range(int(N_predicted / group)):
            orig_rgb_stripe = group_images(test_imgs_orig[i * group:(i * group) + group, :, :, :], group) / 255.
            orig_stripe = group_images(orig_imgs[i * group:(i * group) + group, :, :, :], group)
            pred_stripe = group_images(pred_imgs[i * group:(i * group) + group, :, :, :], group)
            total_img =  np.tile(pred_stripe, 3)
            visualize(total_img,
                      path_experiment + name_experiment + "_RGB_Original_GroundTruth_Prediction_" + files[k])  # .show()
