import numpy as np

np.random.seed(1337)

from utils.help_functions import load_hdf5, visualize, group_images
from utils.pre_processing import my_PreProc


# from keras.preprocessing.image import array_to_img
# Load the original data and return the extracted patches for training/testing
def get_data_testing(test_imgs_original, test_groudTruth, Imgs_to_test, patch_height, patch_width):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_masks = load_hdf5(test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.

    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_masks = test_masks[0:Imgs_to_test, :, :, :]
    test_imgs = paint_border(test_imgs, patch_height, patch_width)
    test_masks = paint_border(test_masks, patch_height, patch_width)

    data_consistency_check(test_imgs, test_masks)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images/masks shape:")
    print(test_imgs.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered(test_imgs, patch_height, patch_width)
    patches_masks_test = extract_ordered(test_masks, patch_height, patch_width)
    data_consistency_check(patches_imgs_test, patches_masks_test)

    print("\ntest PATCHES images/masks shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))
    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1))
    patches_masks_test = np.transpose(test_masks, (0, 2, 3, 1))
    return patches_imgs_test, patches_masks_test


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap_ex(test_imgs_original, Imgs_to_test, patch_height, patch_width,
                                stride_height, stride_width):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)

    test_imgs = my_PreProc(test_imgs_original)
    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    # check masks are within 0-1
    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")
    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)
    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))
    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3]


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap(test_imgs_original, test_groudTruth, patch_height, patch_width,
                             stride_height, stride_width):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_masks = load_hdf5(test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    # extend both images and masks so they can be divided exactly by the patches dimensions
    # test_imgs = test_imgs[15:30, :, :, :]
    # test_masks = test_masks[15:30, :, :, :]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))
    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1))
    test_masks = np.transpose(test_masks, (0, 2, 3, 1))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


# Load the original data and return the extracted patches for testing
# return the ground truth in its original shape
def get_data_testing_overlap_stare(test_imgs_original, test_groudTruth, patch_height, patch_width,
                                   stride_height, stride_width, imgidx=0):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_masks = load_hdf5(test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.
    print('index: ' + str(imgidx))
    print('STARE original shape:')
    test_imgs = test_imgs[imgidx:imgidx + 1, ...]
    test_masks = test_masks[imgidx:imgidx + 1, ...]

    # extend both images and masks so they can be divided exactly by the patches dimensions
    # test_imgs = test_imgs[0:Imgs_to_test, :, :, :]
    # test_masks = test_masks[0:Imgs_to_test, :, :, :]
    test_imgs = paint_border_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images shape:")
    print(test_imgs.shape)
    print("\ntest mask shape:")
    print(test_masks.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_overlap(test_imgs, patch_height, patch_width, stride_height, stride_width)

    print("\ntest PATCHES images shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))
    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1))
    test_masks = np.transpose(test_masks, (0, 2, 3, 1))
    return patches_imgs_test, test_imgs.shape[2], test_imgs.shape[3], test_masks


def get_data_training(train_imgs_original,
                      train_groudTruth,
                      patch_height,
                      patch_width,
                      N_subimgs,
                      inside_FOV,
                      fcn=True):
    train_imgs_original = load_hdf5(train_imgs_original)
    train_masks = load_hdf5(train_groudTruth)  # masks always the same

    # visualize(group_images(train_imgs_original[:, :, :, :], 5),
    #           path_experiment + 'imgs_train')

    # TODO: preprocessing
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.
    # if dataset == 'DRIVE':
    #     train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    #     train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # elif dataset == 'STARE':
    #     train_imgs = train_imgs[:, :, :, 15:685]
    #     train_masks = train_masks[:, :, :, 15:685]
    # elif dataset == 'CHASE':
    #     train_imgs = train_imgs[:, :, :, 19:979]
    #     train_masks = train_masks[:, :, :, 19:979]
    # elif dataset == 'HRF':
    #     train_imgs = train_imgs[:, :, :, 19:979]
    #     train_masks = train_masks[:, :, :, 19:979]
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    if fcn:
        patches_imgs_train, patches_masks_train = extract_random_patches(train_imgs, train_masks,
                                                                         patch_height, patch_width,
                                                                         N_subimgs,
                                                                         inside=inside_FOV)
        data_consistency_check(patches_imgs_train, patches_masks_train)
    else:
        patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks,
                                                                 patch_height, patch_width,
                                                                 N_subimgs,
                                                                 inside=inside_FOV)
    ##Fourier transform of patches
    # TODO add hessian grangi
    # patches_imgs_train = my_PreProc_patches(patches_imgs_train)
    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train  # , patches_imgs_test, patches_masks_test


def get_data_training_stare(train_imgs_original,
                            train_groudTruth,
                            patch_height,
                            patch_width,
                            N_subimgs,
                            inside_FOV, imgidx=0,
                            fcn=True):
    train_imgs_original = load_hdf5(train_imgs_original)
    train_masks = load_hdf5(train_groudTruth)  # masks always the same
    print('index: ' + str(imgidx))
    print('STARE original shape:')
    print(train_imgs_original.shape)
    train_imgs_original = np.delete(train_imgs_original, imgidx, axis=0)
    train_masks = np.delete(train_masks, imgidx, axis=0)
    print('STARE deleted ' + str(imgidx) + 'shape')
    print(train_imgs_original.shape)
    # visualize(group_images(train_imgs_original[:, :, :, :], 5),
    #           path_experiment + 'imgs_train')

    # TODO: preprocessing
    train_imgs = my_PreProc(train_imgs_original)
    train_masks = train_masks / 255.
    # if dataset == 'DRIVE':
    #     train_imgs = train_imgs[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    #     train_masks = train_masks[:, :, 9:574, :]  # cut bottom and top so now it is 565*565
    # elif dataset == 'STARE':
    #     train_imgs = train_imgs[:, :, :, 15:685]
    #     train_masks = train_masks[:, :, :, 15:685]
    # elif dataset == 'CHASE':
    #     train_imgs = train_imgs[:, :, :, 19:979]
    #     train_masks = train_masks[:, :, :, 19:979]
    # elif dataset == 'HRF':
    #     train_imgs = train_imgs[:, :, :, 19:979]
    #     train_masks = train_masks[:, :, :, 19:979]
    data_consistency_check(train_imgs, train_masks)

    # check masks are within 0-1
    assert (np.min(train_masks) == 0 and np.max(train_masks) == 1)

    print("\ntrain images/masks shape:")
    print(train_imgs.shape)
    print("train images range (min-max): " + str(np.min(train_imgs)) + ' - ' + str(np.max(train_imgs)))
    print("train masks are within 0-1\n")

    # extract the TRAINING patches from the full images
    if fcn:
        patches_imgs_train, patches_masks_train = extract_random_patches(train_imgs, train_masks,
                                                                         patch_height, patch_width,
                                                                         N_subimgs,
                                                                         inside=inside_FOV)
        data_consistency_check(patches_imgs_train, patches_masks_train)
    else:
        patches_imgs_train, patches_masks_train = extract_random(train_imgs, train_masks,
                                                                 patch_height, patch_width,
                                                                 N_subimgs,
                                                                 inside=inside_FOV)
    ##Fourier transform of patches
    # TODO add hessian grangi
    # patches_imgs_train = my_PreProc_patches(patches_imgs_train)
    print("\ntrain PATCHES images/masks shape:")
    print(patches_imgs_train.shape)
    print("train PATCHES images range (min-max): " + str(np.min(patches_imgs_train)) + ' - ' + str(
        np.max(patches_imgs_train)))

    return patches_imgs_train, patches_masks_train  # , patches_imgs_test, patches_masks_test


# extract patches randomly in the full training images
#  -- Inside OR in full image balance
def extract_random(full_imgs, full_masks, patch_h, patch_w, N_patches,
                   inside=True):
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of ", full_imgs.shape[0])
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays

    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3

    # assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    # patches = np.empty((N_patches,full_imgs.shape[1],resize_tuple[1],resize_tuple[2]))
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty(N_patches)
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)

    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x_center = np.random.randint(low=0 + int(patch_w / 2), high=img_w - int(patch_w / 2))
            # print "x_center " +str(x_center)
            y_center = np.random.randint(low=0 + int(patch_h / 2), high=img_h - int(patch_h / 2))
            # print "y_center " +str(y_center)
            # check whether the patch is fully contained in the FOV
            if inside == True:
                # only for drive
                if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h) == False:
                    continue
                    # print "y_center " +str(y_center)
                    # check whether the patch is fully contained in the FOV
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2) + 1,
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2) + 1]
            patch_mask = full_masks[i, 0, y_center, x_center]

            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1  # total
            k += 1  # per full_img
    from keras.utils import np_utils
    patches_masks = np_utils.to_categorical(patches_masks, 2)
    return patches, patches_masks


# extract patches randomly in the full validation images
#  -- Inside OR in full image
def extract_random_patches(full_imgs, full_masks, patch_h, patch_w, N_patches, inside=True):
    if (N_patches % full_imgs.shape[0] != 0):
        print("N_patches: plase enter a multiple of ", full_imgs.shape[0])
        exit()
    assert (len(full_imgs.shape) == 4 and len(full_masks.shape) == 4)  # 4D arrays

    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3

    # assert (full_masks.shape[1]==1)   #masks only black and white
    assert (full_imgs.shape[2] == full_masks.shape[2] and full_imgs.shape[3] == full_masks.shape[3])
    # patches = np.empty((N_patches,full_imgs.shape[1],resize_tuple[1],resize_tuple[2]))
    patches = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    patches_masks = np.empty((N_patches, full_imgs.shape[1], patch_h, patch_w))
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    # (0,0) in the center of the image
    patch_per_img = int(N_patches / full_imgs.shape[0])  # N_patches equally divided in the full images
    print("patches per full image: " + str(patch_per_img))
    iter_tot = 0  # iter over the total numbe rof patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        k = 0
        while k < patch_per_img:
            x_center = np.random.randint(low=0 + int(patch_w / 2), high=img_w - int(patch_w / 2))
            # print "x_center " +str(x_center)
            y_center = np.random.randint(low=0 + int(patch_h / 2), high=img_h - int(patch_h / 2))
            # print "y_center " +str(y_center)
            # check whether the patch is fully contained in the FOV
            # if inside == True:
            # only for drive
            # if is_patch_inside_FOV(x_center, y_center, img_w, img_h, patch_h) == False:
            #     continue
            patch = full_imgs[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                    x_center - int(patch_w / 2):x_center + int(patch_w / 2)]
            patch_mask = full_masks[i, :, y_center - int(patch_h / 2):y_center + int(patch_h / 2),
                         x_center - int(patch_w / 2):x_center + int(patch_w / 2)]

            patches[iter_tot] = patch
            patches_masks[iter_tot] = patch_mask
            iter_tot += 1  # total
            k += 1  # per full_img

    return patches, patches_masks


# Divide all the full_imgs in pacthes
def extract_ordered(full_imgs, patch_h, patch_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    N_patches_h = int(img_h / patch_h)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_h) + " patches in height, with about " + str(
            img_h % patch_h) + " pixels left over")
    N_patches_w = int(img_w / patch_w)  # round to lowest int
    if (img_h % patch_h != 0):
        print("warning: " + str(N_patches_w) + " patches in width, with about " + str(
            img_w % patch_w) + " pixels left over")
    print("number of patches per image: " + str(N_patches_h * N_patches_w))
    N_patches_tot = (N_patches_h * N_patches_w) * full_imgs.shape[0]
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range(N_patches_h):
            for w in range(N_patches_w):
                patch = full_imgs[i, :, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


def data_consistency_check(imgs, masks):
    assert (len(imgs.shape) == len(masks.shape))
    assert (imgs.shape[0] == masks.shape[0])
    assert (imgs.shape[2] == masks.shape[2])
    assert (imgs.shape[3] == masks.shape[3])
    assert (masks.shape[1] == 1)

    # image.shape[1] >1 in using gabor wavelet,  so cannot have fixed number of channels
    # assert(imgs.shape[1]==1 or imgs.shape[1]==3)


def is_patch_inside_FOV(x, y, img_w, img_h, patch_h):
    x_ = x - int(img_w / 2)  # origin (0,0) shifted to image center
    y_ = y - int(img_h / 2)  # origin (0,0) shifted to image center
    R_inside = 270 - int(patch_h * np.sqrt(
        2.0) / 2.0)  # radius is 270 (from DRIVE db docs), minus the patch diagonal (assumed it is a square #this is the limit to contain the full patch in the FOV
    radius = np.sqrt((x_ * x_) + (y_ * y_))
    if radius < R_inside:
        return True
    else:
        return False


def recompone_overlap(preds, img_h, img_w, stride_h, stride_w):
    assert (len(preds.shape) == 4)  # 4D arrays
    # assert (preds.shape[1]==1 or preds.shape[1]==3)  #check the channel is 1 or 3
    patch_h = preds.shape[2]
    patch_w = preds.shape[3]
    N_patches_h = (img_h - patch_h) // stride_h + 1
    N_patches_w = (img_w - patch_w) // stride_w + 1
    N_patches_img = N_patches_h * N_patches_w
    print("N_patches_h: " + str(N_patches_h))
    print("N_patches_w: " + str(N_patches_w))
    print("N_patches_img: " + str(N_patches_img))
    # assert (preds.shape[0]%N_patches_img==0)
    N_full_imgs = preds.shape[0] // N_patches_img
    print("According to the dimension inserted, there are " + str(N_full_imgs) + " full images (of " + str(
        img_h) + "x" + str(img_w) + " each)")
    full_prob = np.zeros(
        (N_full_imgs, preds.shape[1], img_h, img_w))  # itialize to zero mega array with sum of Probabilities
    full_sum = np.zeros((N_full_imgs, preds.shape[1], img_h, img_w))
    k = 0  # iterator over all the patches
    for i in range(N_full_imgs):
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                full_prob[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += \
                    preds[k]
                full_sum[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w] += 1
                k += 1
    print(k, preds.shape[0])
    assert (k == preds.shape[0])
    assert (np.min(full_sum) >= 1.0)  # at least one
    final_avg = full_prob / full_sum
    print('using avg')
    # assert(np.max(final_avg)<=1.0) #max value for a pixel is 1.0
    # assert(np.min(final_avg)>=0.0) #min value for a pixel is 0.0
    return final_avg


# Recompone the full images with the patches
def recompone(data, N_h, N_w):
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    assert (len(data.shape) == 4)
    N_pacth_per_img = N_w * N_h
    assert (data.shape[0] % N_pacth_per_img == 0)
    N_full_imgs = data.shape[0] / N_pacth_per_img
    patch_h = data.shape[2]
    patch_w = data.shape[3]
    N_pacth_per_img = N_w * N_h
    # define and start full recompone
    full_recomp = np.empty((N_full_imgs, data.shape[1], N_h * patch_h, N_w * patch_w))
    k = 0  # iter full img
    s = 0  # iter single patch
    while (s < data.shape[0]):
        # recompone one:
        single_recon = np.empty((data.shape[1], N_h * patch_h, N_w * patch_w))
        for h in range(N_h):
            for w in range(N_w):
                single_recon[:, h * patch_h:(h * patch_h) + patch_h, w * patch_w:(w * patch_w) + patch_w] = data[s]
                s += 1
        full_recomp[k] = single_recon
        k += 1
    assert (k == N_full_imgs)
    return full_recomp


def paint_border_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    leftover_h = (img_h - patch_h) % stride_h  # leftover on the h dim
    leftover_w = (img_w - patch_w) % stride_w  # leftover on the w dim
    if (leftover_h != 0):  # change dimension of img_h
        print("\nthe side H is not compatible with the selected stride of " + str(stride_h))
        print("img_h " + str(img_h) + ", patch_h " + str(patch_h) + ", stride_h " + str(stride_h))
        print("(img_h - patch_h) MOD stride_h: " + str(leftover_h))
        print("So the H dim will be padded with additional " + str(stride_h - leftover_h) + " pixels")
        tmp_full_imgs = np.zeros((full_imgs.shape[0], full_imgs.shape[1], img_h + (stride_h - leftover_h), img_w))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:img_h, 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    if (leftover_w != 0):  # change dimension of img_w
        print("the side W is not compatible with the selected stride of " + str(stride_w))
        print("img_w " + str(img_w) + ", patch_w " + str(patch_w) + ", stride_w " + str(stride_w))
        print("(img_w - patch_w) MOD stride_w: " + str(leftover_w))
        print("So the W dim will be padded with additional " + str(stride_w - leftover_w) + " pixels")
        tmp_full_imgs = np.zeros(
            (full_imgs.shape[0], full_imgs.shape[1], full_imgs.shape[2], img_w + (stride_w - leftover_w)))
        tmp_full_imgs[0:full_imgs.shape[0], 0:full_imgs.shape[1], 0:full_imgs.shape[2], 0:img_w] = full_imgs
        full_imgs = tmp_full_imgs
    print("new full images shape: \n" + str(full_imgs.shape))
    return full_imgs


# Extend the full images becasue patch divison is not exact
def paint_border(data, patch_h, patch_w):
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    img_h = data.shape[2]
    img_w = data.shape[3]
    new_img_h = 0
    new_img_w = 0
    if (img_h % patch_h) == 0:
        new_img_h = img_h
    else:
        new_img_h = ((int(img_h) / int(patch_h)) + 1) * patch_h
    if (img_w % patch_w) == 0:
        new_img_w = img_w
    else:
        new_img_w = ((int(img_w) / int(patch_w)) + 1) * patch_w
    new_data = np.zeros((data.shape[0], data.shape[1], new_img_h, new_img_w))
    new_data[:, :, 0:img_h, 0:img_w] = data[:, :, :, :]
    return new_data


# Divide all the full_imgs in pacthes
def extract_ordered_overlap(full_imgs, patch_h, patch_w, stride_h, stride_w):
    assert (len(full_imgs.shape) == 4)  # 4D arrays
    assert (full_imgs.shape[1] == 1 or full_imgs.shape[1] == 3)  # check the channel is 1 or 3
    img_h = full_imgs.shape[2]  # height of the full image
    img_w = full_imgs.shape[3]  # width of the full image
    assert ((img_h - patch_h) % stride_h == 0 and (img_w - patch_w) % stride_w == 0)
    N_patches_img = ((img_h - patch_h) // stride_h + 1) * (
            (img_w - patch_w) // stride_w + 1)  # // --> division between integers
    N_patches_tot = N_patches_img * full_imgs.shape[0]
    print("Number of patches on h : " + str(((img_h - patch_h) // stride_h + 1)))
    print("Number of patches on w : " + str(((img_w - patch_w) // stride_w + 1)))
    print("number of patches per image: " + str(N_patches_img) + ", totally for this dataset: " + str(N_patches_tot))
    patches = np.empty((N_patches_tot, full_imgs.shape[1], patch_h, patch_w))
    iter_tot = 0  # iter over the total number of patches (N_patches)
    for i in range(full_imgs.shape[0]):  # loop over the full images
        for h in range((img_h - patch_h) // stride_h + 1):
            for w in range((img_w - patch_w) // stride_w + 1):
                patch = full_imgs[i, :, h * stride_h:(h * stride_h) + patch_h, w * stride_w:(w * stride_w) + patch_w]
                patches[iter_tot] = patch
                iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches


# return only the pixels contained in the FOV, for both images and masks
def pred_only_FOV(data_imgs, data_masks, original_imgs_border_masks=None, insideFOV=True):
    assert (len(data_imgs.shape) == 4 and len(data_masks.shape) == 4)  # 4D arrays
    assert (data_imgs.shape[0] == data_masks.shape[0])
    assert (data_imgs.shape[2] == data_masks.shape[2])
    assert (data_imgs.shape[3] == data_masks.shape[3])
    assert (data_imgs.shape[1] == 1 and data_masks.shape[1] == 1)  # check the channel is 1
    height = data_imgs.shape[2]
    width = data_imgs.shape[3]
    new_pred_imgs = []
    new_pred_masks = []
    for i in range(data_imgs.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if insideFOV:
                    if inside_FOV(i, x, y, original_imgs_border_masks) == True:
                        new_pred_imgs.append(data_imgs[i, :, y, x])
                        new_pred_masks.append(data_masks[i, :, y, x])
                else:
                    new_pred_imgs.append(data_imgs[i, :, y, x])
                    new_pred_masks.append(data_masks[i, :, y, x])
    new_pred_imgs = np.asarray(new_pred_imgs)
    new_pred_masks = np.asarray(new_pred_masks)
    new_pred_imgs = np.reshape(new_pred_imgs, (new_pred_imgs.shape[0]))
    new_pred_masks = np.reshape(new_pred_masks, (new_pred_masks.shape[0]))
    return new_pred_imgs, new_pred_masks


# function to set to black everything outside the FOV, in a full image
def kill_border(data, original_imgs_border_masks):
    assert (len(data.shape) == 4)  # 4D arrays
    assert (data.shape[1] == 1 or data.shape[1] == 3)  # check the channel is 1 or 3
    height = data.shape[2]
    width = data.shape[3]
    for i in range(data.shape[0]):  # loop over the full images
        for x in range(width):
            for y in range(height):
                if inside_FOV(i, x, y, original_imgs_border_masks) == False:
                    data[i, :, y, x] = 0.0


def inside_FOV(i, x, y, masks):
    assert (len(masks.shape) == 4)  # 4D arrays
    assert (masks.shape[1] == 1)  # DRIVE masks is black and white
    # DRIVE_masks = DRIVE_masks/255.  #NOOO!! otherwise with float numbers takes forever!!

    if (x >= masks.shape[3] or y >= masks.shape[2]):  # my image bigger than the original
        return False

    if (masks[i, 0, y, x] > 0):  # 0==black pixels
        # print DRIVE_masks[i,0,y,x]  #verify it is working right
        return True
    else:
        return False


# Load the original data and return the extracted patches for training/testing
def get_data_testing_single_image(test_imgs_original,
                                  test_groudTruth,
                                  patch_height,
                                  patch_width,
                                  index):
    ### test
    test_imgs_original = load_hdf5(test_imgs_original)
    test_masks = load_hdf5(test_groudTruth)

    test_imgs = my_PreProc(test_imgs_original)
    test_masks = test_masks / 255.

    # extend both images and masks so they can be divided exactly by the patches dimensions
    test_imgs = test_imgs[index, :, :, :]
    test_masks = test_masks[index, :, :, :]
    # test_imgs = paint_border(test_imgs, patch_height, patch_width)
    # test_masks = paint_border(test_masks, patch_height, patch_width)

    # check masks are within 0-1
    assert (np.max(test_masks) == 1 and np.min(test_masks) == 0)

    print("\ntest images/masks shape:")
    print(test_imgs.shape)
    print("test images range (min-max): " + str(np.min(test_imgs)) + ' - ' + str(np.max(test_imgs)))
    print("test masks are within 0-1\n")

    # extract the TEST patches from the full images
    patches_imgs_test = extract_ordered_single_image(test_imgs, patch_height, patch_width)

    print("\ntest PATCHES images/masks shape:")
    print(patches_imgs_test.shape)
    print("test PATCHES images range (min-max): " + str(np.min(patches_imgs_test)) + ' - ' + str(
        np.max(patches_imgs_test)))
    patches_imgs_test = np.transpose(patches_imgs_test, (0, 2, 3, 1))
    return patches_imgs_test


# Divide all the full_img in pacthes
def extract_ordered_single_image(full_img, patch_h, patch_w):
    assert (len(full_img.shape) == 3)  # 3D arrays
    assert (full_img.shape[0] == 1 or full_img.shape[0] == 3)  # check the channel is 1 or 3
    img_h = full_img.shape[1]  # height of the full image
    img_w = full_img.shape[2]  # width of the full image
    N_patches_h = img_h - (patch_h - 1)
    N_patches_w = img_w - (patch_w - 1)

    N_patches_tot = (N_patches_h * N_patches_w)

    patches = np.empty((N_patches_tot, full_img.shape[0], patch_h, patch_w))

    iter_tot = 0  # iter over the total number of patches (N_patches)
    for h in range(N_patches_h):
        for w in range(N_patches_w):
            patch = full_img[:, h:(h + patch_h), w:(w + patch_w)]
            patches[iter_tot] = patch
            iter_tot += 1  # total
    assert (iter_tot == N_patches_tot)
    return patches  # array with all the full_imgs divided in patches
