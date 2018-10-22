import h5py
import numpy as np
import keras.backend as K

np.random.seed(1337)
from PIL import Image


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:  # "with" close the file after its nested commands
        return f["image"][()]


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


# convert RGB image in black and white
def rgb2gray(rgb):
    assert (len(rgb.shape) == 4)  # 4D arrays
    assert (rgb.shape[1] == 3)
    bn_imgs = rgb[:, 0, :, :] * 0.299 + rgb[:, 1, :, :] * 0.587 + rgb[:, 2, :, :] * 0.114
    bn_imgs = np.reshape(bn_imgs, (rgb.shape[0], 1, rgb.shape[2], rgb.shape[3]))
    return bn_imgs


# visualize image (as PIL image, NOT as matplotlib!)
def visualize(data, filename):
    assert (len(data.shape) == 3)  # height*width*channels
    img = None
    if data.shape[2] == 1:  # in case it is black and white
        data = np.reshape(data, (data.shape[0], data.shape[1]))
    if np.max(data) > 1:
        img = Image.fromarray(data.astype(np.uint8))  # the image is already 0-255
    else:
        img = Image.fromarray((data * 255).astype(np.uint8))  # the image is between 0-1
    img.save(filename + '.png')
    return img


# group a set of images row per columns
def group_images(data, per_row):
    assert data.shape[0] % per_row == 0
    assert (data.shape[1] == 1 or data.shape[1] == 3)
    data = np.transpose(data, (0, 2, 3, 1))  # corect format for imshow
    all_stripe = []
    for i in range(int(data.shape[0] / per_row)):
        stripe = data[i * per_row]
        for k in range(i * per_row + 1, i * per_row + per_row):
            stripe = np.concatenate((stripe, data[k]), axis=1)
        all_stripe.append(stripe)
    totimg = all_stripe[0]
    for i in range(1, len(all_stripe)):
        totimg = np.concatenate((totimg, all_stripe[i]), axis=0)
    return totimg


def conv_to_imgs(pred, img_h, img_w, patch_h, patch_w, path_experiment, index, mode='original'):
    assert (len(pred.shape) == 2)  # 3D array: (Npatches,2)
    assert (pred.shape[1] == 2)  # check the classes are 2
    pred_image = np.empty((pred.shape[0]))  # (Npatches,height*width)
    pred_image_none = np.empty((pred.shape[0]))  # (Npatches,height*width)
    img_descp = mode
    threshold = 0.5
    if mode == "original":
        for i in range(pred.shape[0]):
            pred_image[i] = pred[i, 1]
            # pred_image_none[i]=pred[i, 0]
    elif mode == "threshold":
        img_descp += "_" + str(threshold)
        for i in range(pred.shape[0]):
            if pred[i, 1] >= threshold:
                pred_image[i] = 1
            else:
                pred_image[i] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    img_descp += "[" + str(index) + "]"
    pred_image = np.reshape(pred_image, (1, (img_h - (patch_h - 1)), (img_w - (patch_w - 1))))
    final_image = np.zeros((1, img_h, img_w))
    final_image[:, int((patch_h - 1) / 2):int(img_h - (patch_h - 1) / 2),
    int((patch_w - 1) / 2):int(img_w - (patch_w - 1) / 2)] = pred_image
    print(pred_image.shape, final_image.shape)
    visualize(np.transpose(final_image, (1, 2, 0)), path_experiment + 'test_prediction_' + img_descp)

    # pred_image_none = np.reshape(pred_image_none, (1, (img_h - (patch_h - 1)), (img_w - (patch_w - 1))))
    # final_image = np.zeros((1, img_h, img_w))
    # final_image[:, int((patch_h - 1) / 2):int(img_h - (patch_h - 1) / 2),
    # int((patch_w - 1) / 2):int(img_w - (patch_w - 1) / 2)] = pred_image_none
    # print(pred_image_none.shape, final_image.shape)
    # visualize(np.transpose(final_image, (1, 2, 0)), path_experiment + 'test_prediction_none' + img_descp)

    return final_image


def pred_to_imgs(pred, patch_height, patch_width, mode="original"):
    assert (len(pred.shape) == 4)  # 3D array: (Npatches,height*width,2)
    assert (pred.shape[2] == 2)  # check the classes are 2
    pred_images = np.empty((pred.shape[0], pred.shape[1]))  # (Npatches,height*width)
    if mode == "original":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                pred_images[i, pix] = pred[i, pix, 1]
    elif mode == "threshold":
        for i in range(pred.shape[0]):
            for pix in range(pred.shape[1]):
                if pred[i, pix, 1] >= 0.5:
                    pred_images[i, pix] = 1
                else:
                    pred_images[i, pix] = 0
    else:
        print("mode " + str(mode) + " not recognized, it can be 'original' or 'threshold'")
        exit()
    pred_images = np.reshape(pred_images, (pred_images.shape[0], 1, patch_height, patch_width))
    return pred_images


def dense_crf(img, output_probs):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import unary_from_softmax
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    U = -np.log(output_probs)
    U = U.reshape((2, -1))
    U = np.ascontiguousarray(U)
    # U = unary_from_softmax(output_probs)
    img = np.ascontiguousarray(img)
    U = U.astype(np.float32)
    d.setUnaryEnergy(U)

    # This adds the color-independent term, features are the locations only.
    d.addPairwiseGaussian(sxy=0.1, compat=0.1, kernel=dcrf.DIAG_KERNEL, normalization=dcrf.NORMALIZE_SYMMETRIC)

    # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
    d.addPairwiseBilateral(sxy=0.1, srgb=0.1, rgbim=img.astype(np.uint8), compat=10, kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(10)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))

    return Q

def dense_crf_no_rgb(img, output_probs):
    import pydensecrf.densecrf as dcrf
    from pydensecrf.utils import create_pairwise_bilateral,unary_from_softmax
    h = output_probs.shape[0]
    w = output_probs.shape[1]

    output_probs = np.expand_dims(output_probs, 0)
    output_probs = np.append(1 - output_probs, output_probs, axis=0)

    d = dcrf.DenseCRF2D(w, h, 2)
    # U = -np.log(output_probs)
    # U = U.reshape((2, -1))
    # U = np.ascontiguousarray(U)
    U = unary_from_softmax(output_probs)
    pairwise_energy = create_pairwise_bilateral(sdims=(1, 1), schan=(0.01,), img=img)
    U = U.astype(np.float32)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(5):
        d.stepInference(Q, tmp1, tmp2)
    Q = np.argmax(np.array(Q), axis=0).reshape((h, w))
    return Q


def get_layer_outputs(test_image, model):
    outputs = [layer.output for layer in model.layers]  # all layer outputs
    comp_graph = [K.function([model.input] + [K.learning_phase()], [output]) for output in
                  outputs]  # evaluation functions

    # Testing
    layer_outputs_list = [op([test_image, 1.]) for op in comp_graph]
    layer_outputs = []

    for layer_output in layer_outputs_list:
        # print(layer_output[0][0].shape, end='\n-------------------\n')
        layer_outputs.append(layer_output[0][0])

    return layer_outputs


def plot_layer_outputs(test_image, layer_number, model, c=3):
    layer_outputs = get_layer_outputs(test_image, model)

    x_max = layer_outputs[layer_number].shape[0]
    y_max = layer_outputs[layer_number].shape[1]
    n = layer_outputs[layer_number].shape[2]

    L = []
    for i in range(n):
        L.append(np.zeros((x_max, y_max)))

    for i in range(n):
        for x in range(x_max):
            for y in range(y_max):
                L[i][x][y] = layer_outputs[layer_number][x][y][i]

    # for img in L:
    #     plt.figure()
    #     plt.imshow(img, interpolation='nearest')
    return L
