from keras import backend as K
from keras import regularizers
from keras.layers import Input, merge, concatenate, Concatenate, Reshape, ZeroPadding2D, GlobalAvgPool2D, add
from keras.layers.convolutional import Conv2D, Convolution2D, MaxPooling2D, UpSampling2D, \
    UpSampling2D, AveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.models import Sequential
from utils.layers import ConvOffset2D
from keras.optimizers import *


def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def build_2d_unet_model(inp_shape, k_size=3, downsize_nb_filters_factor=2):
    merge_axis = -1  # Feature maps are concatenated along last axis (for tf backend)
    data = Input(shape=inp_shape)

    conv1 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)
    conv4 = Dropout(0.5)(conv4)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(int(256 / downsize_nb_filters_factor), k_size, padding='same')(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = Conv2D(int(256 / downsize_nb_filters_factor), k_size, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)
    conv6 = Dropout(0.5)(conv6)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(int(512 / downsize_nb_filters_factor), k_size, padding='same')(pool3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('relu')(conv7)

    conv8 = Conv2D(int(512 / downsize_nb_filters_factor), k_size, padding='same')(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('relu')(conv8)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    conv9 = Conv2D(int(1024 / downsize_nb_filters_factor), k_size, padding='same')(pool4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('relu')(conv9)

    conv10 = Conv2D(int(1024 / downsize_nb_filters_factor), k_size, padding='same')(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation('relu')(conv10)

    conv11 = Conv2D(int(512 / downsize_nb_filters_factor), k_size, padding='same')(conv10)
    conv11 = BatchNormalization()(conv11)
    conv11 = Activation('relu')(conv11)

    up1 = UpSampling2D(size=(2, 2))(conv11)
    merged1 = concatenate([up1, conv8], axis=merge_axis)

    conv12 = Conv2D(int(512 / downsize_nb_filters_factor), k_size, padding='same')(merged1)
    conv12 = BatchNormalization()(conv12)
    conv12 = Activation('relu')(conv12)

    conv13 = Conv2D(int(512 / downsize_nb_filters_factor), k_size, padding='same')(conv12)
    conv13 = BatchNormalization()(conv13)
    conv13 = Activation('relu')(conv13)

    up2 = UpSampling2D(size=(2, 2))(conv13)
    merged2 = concatenate([up2, conv6], axis=merge_axis)

    conv14 = Conv2D(int(256 / downsize_nb_filters_factor), k_size, padding='same')(merged2)
    conv14 = BatchNormalization()(conv14)
    conv14 = Activation('relu')(conv14)

    conv15 = Conv2D(int(256 / downsize_nb_filters_factor), k_size, padding='same')(conv14)
    conv15 = BatchNormalization()(conv15)
    conv15 = Activation('relu')(conv15)

    up3 = UpSampling2D(size=(2, 2))(conv15)
    merged3 = concatenate([up3, conv4], axis=merge_axis)
    conv16 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(merged3)
    conv16 = BatchNormalization()(conv16)
    conv16 = Activation('relu')(conv16)

    conv17 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(conv16)
    conv17 = BatchNormalization()(conv17)
    conv17 = Activation('relu')(conv17)

    up4 = UpSampling2D(size=(2, 2))(conv17)
    merged4 = concatenate([up4, conv2], axis=merge_axis)

    conv18 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(merged4)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)

    conv18 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(conv18)
    conv18 = BatchNormalization()(conv18)
    conv18 = Activation('relu')(conv18)

    output = Conv2D(1, 1, padding='same', activation='sigmoid')(conv18)
    # conv19 = Conv2D(2, k_size, padding='same')(conv18)
    # output = Reshape([-1, 2])(conv19)
    # output = Activation('sigmoid')(output)
    # output = Reshape(inp_shape[:-1] + (2,))(output)
    model = Model(data, output)
    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', dice_coef])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def deform_block(filter, x, name='conv_offset', prior_pool=False, prior_upsampling=False, norm=True, activate=True):
    if prior_pool:
        conv = ConvOffset2D(int(filter / 2), name=name)(x)
    elif prior_upsampling:
        conv = Conv2D(filter, (3, 3), padding='same')(x)
        conv = ConvOffset2D(filter, name=name)(conv)
    else:
        conv = ConvOffset2D(filter, name=name)(x)
    conv = Conv2D(filter, (3, 3), padding='same')(conv)
    if norm:
        conv = BatchNormalization()(conv)
    if activate:
        conv = Activation('relu')(conv)
    return conv


def build_deform_unet(inp_shape, downsize_nb_filters_factor=2, nb_pool=(2, 2), ass_crf=False):
    merge_axis = -1
    inputs = Input(shape=inp_shape)  # TensorFlow backend, dim_order
    l = Conv2D(int(64 / downsize_nb_filters_factor), (3, 3), padding='same', name='conv11')(inputs)
    l = Activation('relu', name='conv11_relu')(l)
    l = BatchNormalization(name='conv11_bn')(l)

    conv1 = deform_block(int(64 / downsize_nb_filters_factor), l, name='conv1_offset')
    conv2 = deform_block(int(64 / downsize_nb_filters_factor), conv1, name='conv2_offset')
    pool1 = MaxPooling2D(pool_size=nb_pool)(conv2)

    conv3 = deform_block(int(128 / downsize_nb_filters_factor), pool1, name='conv3_offset', prior_pool=True)
    conv4 = deform_block(int(128 / downsize_nb_filters_factor), conv3, name='conv4_offset')
    pool2 = MaxPooling2D(pool_size=nb_pool)(conv4)

    conv5 = deform_block(int(256 / downsize_nb_filters_factor), pool2, name='conv5_offset', prior_pool=True)
    conv6 = deform_block(int(256 / downsize_nb_filters_factor), conv5, name='conv6_offset')
    pool3 = MaxPooling2D(pool_size=nb_pool)(conv6)

    conv7 = deform_block(int(512 / downsize_nb_filters_factor), pool3, name='conv7_offset', prior_pool=True)
    conv8 = deform_block(int(512 / downsize_nb_filters_factor), conv7, name='conv8_offset')
    pool4 = MaxPooling2D(pool_size=nb_pool)(conv8)

    conv9 = deform_block(int(1024 / downsize_nb_filters_factor), pool4, name='conv9_offset', prior_pool=True)
    conv10 = deform_block(int(1024 / downsize_nb_filters_factor), conv9, name='conv10_offset')
    conv11 = deform_block(int(1024 / downsize_nb_filters_factor), conv10, name='conv11_offset')
    up1 = UpSampling2D(size=nb_pool)(conv11)

    merged1 = concatenate([up1, conv8], axis=merge_axis)

    conv12 = deform_block(int(512 / downsize_nb_filters_factor), merged1, name='conv12_offset', prior_upsampling=True)
    conv13 = deform_block(int(512 / downsize_nb_filters_factor), conv12, name='conv13_offset')
    up2 = UpSampling2D(size=nb_pool)(conv13)

    merged2 = concatenate([up2, conv6], axis=merge_axis)
    conv14 = deform_block(int(256 / downsize_nb_filters_factor), merged2, name='conv14_offset', prior_upsampling=True)
    conv15 = deform_block(int(256 / downsize_nb_filters_factor), conv14, name='conv15_offset')
    up3 = UpSampling2D(size=nb_pool)(conv15)

    merged3 = concatenate([up3, conv4], axis=merge_axis)

    conv16 = deform_block(int(128 / downsize_nb_filters_factor), merged3, name='conv16_offset', prior_upsampling=True)
    conv17 = deform_block(int(128 / downsize_nb_filters_factor), conv16, name='conv17_offset')
    up4 = UpSampling2D(size=nb_pool)(conv17)

    merged4 = concatenate([up4, conv2], axis=merge_axis)

    conv18 = deform_block(int(64 / downsize_nb_filters_factor), merged4, name='conv18_offset', prior_upsampling=True)
    conv19 = deform_block(int(64 / downsize_nb_filters_factor), conv18, name='conv19_offset')

    output = Conv2D(1, 1, padding='same', activation='sigmoid')(conv19)
    if ass_crf:
        from utils.crfrnn_layer import CrfRnnLayer
        output = CrfRnnLayer(image_dims=(inp_shape[0], inp_shape[1]),
                             num_classes=2,
                             theta_alpha=160.,
                             theta_beta=3.,
                             theta_gamma=3.,
                             num_iterations=10,
                             name='crfrnn')([output, inputs])
        model = Model(inputs, output)
        # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # model.compile(optimizer=adam, loss=output.loss_function, metrics=[output.accuracy])
    else:
        model = Model(inputs, output)
    # adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_deform_cnn(inp_shape, trainable=True):
    inputs = l = Input(inp_shape, name='input')

    # conv11
    l = Conv2D(32, (3, 3), padding='same', name='conv11', trainable=trainable)(l)
    l = BatchNormalization(name='conv11_bn')(l)
    l = Activation('relu', name='conv11_relu')(l)

    # conv12
    l_offset = ConvOffset2D(32, name='conv12_offset')(l)
    l = Conv2D(64, (3, 3), padding='same', strides=(2, 2), name='conv12', trainable=trainable)(l_offset)
    l = BatchNormalization(name='conv12_bn')(l)
    l = Activation('relu', name='conv12_relu')(l)

    # conv21
    l_offset = ConvOffset2D(64, name='conv21_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', name='conv21', trainable=trainable)(l_offset)
    l = BatchNormalization(name='conv21_bn')(l)
    l = Activation('relu', name='conv21_relu')(l)

    # conv22
    l_offset = ConvOffset2D(128, name='conv22_offset')(l)
    l = Conv2D(128, (3, 3), padding='same', strides=(2, 2), name='conv22', trainable=trainable)(l_offset)
    l = BatchNormalization(name='conv22_bn')(l)
    l = Activation('relu', name='conv22_relu')(l)

    # out
    l = GlobalAvgPool2D(name='avg_pool')(l)
    l = Dense(2, name='fc1', trainable=trainable)(l)
    outputs = l = Activation('softmax', name='out')(l)
    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# paper impl: fully convolutional neural network based structured prediction approach towards the retinal vessel segmentation
def build_2d_fcn_paper_model(inp_shape, k_size=3, downsize_nb_filters_factor=2):
    data = Input(shape=inp_shape)

    conv1 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(data)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    conv2 = Conv2D(int(64 / downsize_nb_filters_factor), k_size, padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)

    conv4 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('relu')(conv4)

    up1 = UpSampling2D(size=(2, 2))(conv4)

    conv5 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(up1)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('relu')(conv5)

    conv6 = Conv2D(int(128 / downsize_nb_filters_factor), k_size, padding='same')(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('relu')(conv6)

    output = Conv2D(1, 1, padding='same', activation='sigmoid')(conv6)
    model = Model(data, output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# paper R2U-Net impl
def BuildR2UNet(inp_shape, k_size=3, nb_pool=(2, 2)):
    merge_axis = -1

    def RCL_block(out_num_filters, l, pool=False):
        conv1 = Conv2D(out_num_filters, 1, padding='same')
        stack1 = conv1(l)
        stack2 = BatchNormalization()(stack1)
        stack3 = Activation('relu')(stack2)

        conv2 = Conv2D(out_num_filters, k_size, padding='same', kernel_initializer='he_normal')
        stack4 = conv2(stack3)
        stack5 = add([stack1, stack4])
        stack6 = BatchNormalization()(stack5)
        stack7 = Activation('relu')(stack6)

        conv3 = Conv2D(out_num_filters, k_size, padding='same')
        stack8 = conv3(stack7)
        stack9 = add([stack1, stack8])
        stack10 = BatchNormalization()(stack9)
        stack11 = Activation('relu')(stack10)

        conv4 = Conv2D(out_num_filters, k_size, padding='same')
        stack12 = conv4(stack11)
        stack13 = add([stack1, stack12])
        stack14 = BatchNormalization()(stack13)
        stack15 = Activation('relu')(stack14)

        if pool:
            stack16 = MaxPooling2D((2, 2), border_mode='same')(stack15)
            stack17 = Dropout(0.1)(stack16)
        else:
            stack17 = stack15

        return stack17

    # Build Network
    data = Input(shape=inp_shape)

    conv_l = Conv2D(64, k_size, padding='same', activation='relu')
    l = conv_l(data)
    rcl1 = RCL_block(64, l, pool=False)

    rcl2 = MaxPooling2D(pool_size=nb_pool)(rcl1)
    rcl2 = RCL_block(128, rcl2, pool=False)

    rcl3 = MaxPooling2D(pool_size=nb_pool)(rcl2)
    rcl3 = RCL_block(256, rcl3, pool=False)

    rcl4 = MaxPooling2D(pool_size=nb_pool)(rcl3)
    rcl4 = RCL_block(512, rcl4, pool=False)

    rcl5 = MaxPooling2D(pool_size=nb_pool)(rcl4)
    rcl5 = RCL_block(1024, rcl5, pool=False)

    up1 = UpSampling2D(size=nb_pool)(rcl5)
    merged1 = concatenate([up1, rcl4], axis=merge_axis)
    rcl6 = RCL_block(512, merged1, pool=False)

    up2 = UpSampling2D(size=nb_pool)(rcl6)
    merged2 = concatenate([up2, rcl3], axis=merge_axis)
    rcl7 = RCL_block(256, merged2, pool=False)

    up3 = UpSampling2D(size=nb_pool)(rcl7)
    merged3 = concatenate([up3, rcl2], axis=merge_axis)
    rcl8 = RCL_block(128, merged3, pool=False)

    up4 = UpSampling2D(size=nb_pool)(rcl8)
    merged4 = concatenate([up4, rcl1], axis=merge_axis)
    rcl9 = RCL_block(64, merged4, pool=False)

    convout = Conv2D(1, k_size, padding='same',activation='sigmoid')(rcl9)
    model = Model(data, convout)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model
