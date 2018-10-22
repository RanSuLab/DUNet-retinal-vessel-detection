import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras.backend as K

K.set_image_dim_ordering('tf')
# K.set_image_dim_ordering('th')
import time
import sys

sys.path.insert(0, './utils/')
import numpy as np
from build_model import *
from keras.utils.vis_utils import plot_model
from keras.optimizers import *
from keras.callbacks import EarlyStopping, Callback, ReduceLROnPlateau, CSVLogger
from extract_patches import get_data_training

conf = tf.ConfigProto()
conf.gpu_options.allow_growth = True
sess = tf.Session(config=conf)
import configparser

config = configparser.ConfigParser()
np.random.seed(1337)
from keras.utils import multi_gpu_model


class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


# Get neural network
def get_net(inp_shape, algorithm):
    if algorithm == 'deform':
        return build_deform_cnn(inp_shape=inp_shape)
    elif algorithm == 'unet':
        return build_2d_unet_model(inp_shape=inp_shape)
    elif algorithm == 'deform_unet':
        return build_deform_unet(inp_shape=inp_shape)
    elif algorithm == 'fcn_paper':
        return build_2d_fcn_paper_model(inp_shape=inp_shape)
    elif algorithm == 'R2UNet':
        return BuildR2UNet(inp_shape=inp_shape)


def define_log(log_path_experiment, algorithm):
    if not os.path.exists(log_path_experiment):
        print("DIRECTORY Created")
        os.makedirs(log_path_experiment)

    f = open(log_path_experiment + algorithm + '.log', 'a')
    sys.stdout = Tee(sys.stdout, f)


# because the keras bug. if para we must use the origion model to save the shared weights
class ModelCallBackForMultiGPU(Callback):
    def __init__(self, model):
        self.model_to_save = model

    def on_epoch_end(self, epoch, logs=None):
        if epoch % 1 == 0:
            self.model_to_save.save(log_path_experiment + '/model_at_epoch_%05d.hdf5' % epoch)
            print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accs = []

    def on_epoch_end(self, batch, logs={}):
        self.losses.append(logs.get('val_loss'))
        self.accs.append(logs.get('val_acc'))
        print('learning rate:' + str(logs.get('lr')))


class SGDLearningRateTracker(Callback):
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        lr = K.eval(optimizer.lr * (1. / (1. + optimizer.decay * optimizer.iterations)))
        print('\nLR: {:.6f}\n'.format(lr))


# ========= Load settings from Config file
config.read('configuration.txt')
fine_tuning = False
algorithm = config.get('experiment name', 'name')
dataset = config.get('data attributes', 'dataset')
log_path_experiment = './log/experiments/' + algorithm + '/' + dataset + '/'
# log_path_experiment = './log/experiments/STARE/' + algorithm + '/'

# ========= Load settings from Config file
path_data = config.get('data paths', 'path_local')
model_path = config.get('data paths', 'model_path')
# training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))
inp_shape = (int(config.get('data attributes', 'patch_width')), int(config.get('data attributes', 'patch_height')), 1)
gpu = 3
fcn = True
if algorithm == 'deform':
    fcn = False
    model = get_net(inp_shape=inp_shape, algorithm=algorithm)
else:
    fcn = True
    model = get_net(inp_shape=inp_shape, algorithm=algorithm)

define_log(log_path_experiment, algorithm)

patches_imgs_train, patches_masks_train = get_data_training(
    train_imgs_original=path_data + config.get('data paths', 'train_imgs_original'),
    train_groudTruth=path_data + config.get('data paths', 'train_groundTruth'),  # masks
    patch_height=inp_shape[0],
    patch_width=inp_shape[1],
    N_subimgs=int(config.get('training settings', 'N_subimgs')),
    inside_FOV=config.getboolean('training settings', 'inside_FOV'),
    dataset=dataset,
    path_experiment=log_path_experiment,
    fcn=fcn
)

patches_imgs_train = np.transpose(patches_imgs_train, (0, 2, 3, 1))
if fcn:
    patches_masks_train = np.transpose(patches_masks_train, (0, 2, 3, 1))
print("Check: final output of the network:")
print(model.output_shape)
model.summary()
plot_model(model, to_file=log_path_experiment + algorithm + '_model.png',
           show_shapes=True)  # check how the model looks like
json_string = model.to_json()
open(log_path_experiment + algorithm + '_architecture.json', 'w').write(json_string)
checkpointer = ModelCallBackForMultiGPU(model)
early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='auto')
reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', epsilon=0.0001,
                           cooldown=0, min_lr=0)
csv_logger = CSVLogger(log_path_experiment + 'training.csv', append=True)
hist = model.fit(patches_imgs_train, patches_masks_train, epochs=N_epochs, batch_size=batch_size, validation_split=0.2,
                 callbacks=[checkpointer, reduce, csv_logger, early_stopping], verbose=2)

