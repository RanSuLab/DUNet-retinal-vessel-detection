import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
# scikit learn
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score
import sys

sys.path.insert(0, './utils/')
from help_functions import *
from extract_patches import pred_only_FOV
# ========= CONFIG FILE TO READ FROM =======
import configparser

config = configparser.ConfigParser()
config.read('configuration.txt')
# ===========================================
# model name
path_data = config.get('data paths', 'path_local')
algorithm_config = config.get('experiment name', 'name')

dataset = config.get('data attributes', 'dataset')
name_experiment_list = ["deform_v1", "unet", "deform_unet_v1"]
algorithms = ["Deformable-ConvNet", "U-Net", "DUNet"]
test_border_masks = path_data + config.get('data paths', 'test_border_masks')
test_border_masks = load_hdf5(test_border_masks)

index = 0
for name_experiment in name_experiment_list:
    algorithm = algorithms[index]
    path_experiment = './log/experiments/' + name_experiment + '/' + dataset + '/'
    # if algorithm_config != name_experiment:
    #     continue
    # ========== Elaborate and visualize the predicted images ====================

    pred_imgs = None
    orig_imgs = None
    gtruth_masks = None
    # apply the DRIVE masks on the repdictions #set everything outside the FOV to zero!!
    # kill_border(pred_imgs, test_border_masks)  #DRIVE MASK  #only for visualization
    ## back to original dimensions
    if dataset == 'HRF':
        file = h5py.File(path_experiment + '0:15/' + dataset + '_predict_results.h5', 'r')
        gtruth_masks = file['y_gt'][:]
        pred_imgs = file['y_pred'][:]
        orig_imgs = file['x_origin'][:]
        file.close()
        file = h5py.File(path_experiment + '15:30/' + dataset + '_predict_results.h5', 'r')
        gtruth_masks = np.concatenate([gtruth_masks, file['y_gt'][:]], axis=0)
        pred_imgs = np.concatenate([pred_imgs, file['y_pred'][:]], axis=0)
        file.close()
        gtruth_masks = np.where(gtruth_masks > 0, 1, 0)
    else:
        file = h5py.File(path_experiment + dataset + '_predict_results.h5', 'r')
        gtruth_masks = file['y_gt'][:]
        pred_imgs = file['y_pred'][:]
        orig_imgs = file['x_origin'][:]
        file.close()

    # ====== Evaluate the results
    print("\n\n========  Evaluate the results =======================")
    print('\n', name_experiment)
    print(path_experiment)
    # predictions only inside the FOV
    y_scores, y_true = pred_only_FOV(pred_imgs, gtruth_masks, test_border_masks,
                                     insideFOV=True)  # returns data only inside the FOV
    if np.max(y_true) > 1:
        y_true = y_true // np.max(y_true)
    # Area under the ROC curve
    fpr, tpr, thresholds = roc_curve((y_true), y_scores)
    AUC_ROC = roc_auc_score(y_true, y_scores)
    # test_integral = np.trapz(tpr,fpr) #trapz is numpy integration
    print("\nArea under the ROC curve: " + str(AUC_ROC))
    # roc_curve = plt.figure()
    plt.plot(fpr, tpr, '-', label=algorithm + '_' + dataset + '(AUC = %0.4f)' % AUC_ROC)
    plt.title('ROC curve', fontsize=14)
    plt.xlabel("FPR (False Positive Rate)", fontsize=15)
    plt.ylabel("TPR (True Positive Rate)", fontsize=15)
    plt.legend(loc="lower right")
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.savefig(path_experiment + "ROC.png")

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    precision = np.fliplr([precision])[0]  # so the array is increasing (you won't get negative AUC)
    recall = np.fliplr([recall])[0]  # so the array is increasing (you won't get negative AUC)
    AUC_prec_rec = np.trapz(precision, recall)
    print("\nArea under Precision-Recall curve: " + str(AUC_prec_rec))
    # prec_rec_curve = plt.figure()
    # plt.plot(recall, precision, '-', label='Area Under the Curve (AUC = %0.4f)' % AUC_prec_rec)
    # plt.title('Precision - Recall curve')
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")
    # plt.legend(loc="lower right")
    # plt.xticks(fontsize=14)
    # plt.yticks(fontsize=14)
    # plt.savefig(path_experiment + "Precision_recall.png")

    # Confusion matrix
    threshold_confusion = 0.5
    print("\nConfusion matrix:  Costum threshold (for positive) of " + str(threshold_confusion))
    y_pred = np.empty((y_scores.shape[0]))
    for i in range(y_scores.shape[0]):
        if y_scores[i] >= threshold_confusion:
            y_pred[i] = 1
        else:
            y_pred[i] = 0
    confusion = confusion_matrix(y_true, y_pred)
    print(confusion)
    accuracy = 0
    if float(np.sum(confusion)) != 0:
        accuracy = float(confusion[0, 0] + confusion[1, 1]) / float(np.sum(confusion))
    print("Global Accuracy: " + str(accuracy))
    specificity = 0
    if float(confusion[0, 0] + confusion[0, 1]) != 0:
        specificity = float(confusion[0, 0]) / float(confusion[0, 0] + confusion[0, 1])
    print("Specificity: " + str(specificity))
    sensitivity = 0
    if float(confusion[1, 1] + confusion[1, 0]) != 0:
        sensitivity = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[1, 0])
    print("Sensitivity: " + str(sensitivity))
    precision = 0
    if float(confusion[1, 1] + confusion[0, 1]) != 0:
        precision = float(confusion[1, 1]) / float(confusion[1, 1] + confusion[0, 1])
    print("Precision: " + str(precision))

    # Jaccard similarity index
    jaccard_index = jaccard_similarity_score(y_true, y_pred, normalize=True)
    print("\nJaccard similarity score: " + str(jaccard_index))

    # F1 score
    F1_score = f1_score(y_true, y_pred, labels=None, average='binary', sample_weight=None)
    print("\nF1 score (F-measure): " + str(F1_score))
    # Save the results
    file_perf = open(path_experiment + 'performances_new.txt', 'w')
    file_perf.write("Area under the ROC curve: " + str(AUC_ROC)
                    + "\nArea under Precision-Recall curve: " + str(AUC_prec_rec)
                    + "\nJaccard similarity score: " + str(jaccard_index)
                    + "\nF1 score (F-measure): " + str(F1_score)
                    + "\n\nConfusion matrix:"
                    + str(confusion)
                    + "\nACCURACY: " + str(accuracy)
                    + "\nSENSITIVITY: " + str(sensitivity)
                    + "\nSPECIFICITY: " + str(specificity)
                    + "\nPRECISION: " + str(precision)
                    )
    file_perf.close()
    # break
    index = index + 1
plt.savefig('./log/experiments/' + dataset + "_comparative_ROC.png")
# plt.savefig('./log/experiments/' + dataset + "_Precision_recall.png")

plt.show()
