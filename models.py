#https://github.com/tommyfms2/pix2pix-keras-byt
#http://toxweblog.toxbe.com/2017/12/24/keras-%e3%81%a7-pix2pix-%e3%82%92%e5%ae%9f%e8%a3%85/

from keras.models import Model
from keras.layers.core import Flatten, Dense, Dropout, Activation, Lambda, Reshape
from keras.layers.convolutional import Conv2D, Deconv2D, ZeroPadding2D, UpSampling2D
from keras.layers import Input, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
import keras.backend as K
import numpy as np


def conv_block_unet(x, f, name, bn_axis, bn=True, strides=(2,2)):
    x = LeakyReLU(0.2)(x)
    x = Conv2D(f, (3,3), strides=strides, name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    return x


def up_conv_block_unet(x, x2, f, name, bn_axis, bn=True, dropout=False):
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3,3), name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    if dropout: x = Dropout(0.5)(x)
    x = Concatenate(axis=bn_axis)([x, x2])
    return x

def up_conv_block_unet_alt(x, x2, f, name, bn_axis, bn=True, dropout=False):
    x = Activation('relu')(x)
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(f, (3,3), name=name, padding='same')(x)
    if bn: x = BatchNormalization(axis=bn_axis)(x)
    if dropout: x = Dropout(0.5)(x)
    #x = Concatenate(axis=bn_axis)([x, x2])
    return x


def generator_unet_upsampling(img_shape, disc_img_shape, model_name="generator_unet_upsampling"):
    filters_num = 64
    axis_num = -1
    channels_num = img_shape[-1]
    min_s = min(img_shape[:-1])

    unet_input = Input(shape=img_shape, name="unet_input")

    conv_num = int(np.floor(np.log(min_s)/np.log(2)))
    list_filters_num = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # Encoder
    first_conv = Conv2D(list_filters_num[0], (3,3), strides=(2,2), name='unet_conv2D_1', padding='same')(unet_input)
    list_encoder = [first_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = 'unet_conv2D_' + str(i+2)
        conv = conv_block_unet(list_encoder[-1], f, name, axis_num)
        list_encoder.append(conv)

    # prepare decoder filters
    list_filters_num = list_filters_num[:-2][::-1]
    if len(list_filters_num) < conv_num-1:
        list_filters_num.append(filters_num)

    # Decoder
    first_up_conv = up_conv_block_unet(list_encoder[-1], list_encoder[-2],
                        list_filters_num[0], "unet_upconv2D_1", axis_num, dropout=True)
    list_decoder = [first_up_conv]
    for i, f in enumerate(list_filters_num[1:]):
        name = "unet_upconv2D_" + str(i+2)
        if i<2:  #2
            d = True
            #up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        else:
            d = False
            #up_conv = up_conv_block_unet_alt(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        up_conv = up_conv_block_unet_alt(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        list_decoder.append(up_conv)

    x = Activation('relu')(list_decoder[-1])
    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(disc_img_shape[-1], (3,3), name="last_conv", padding='same')(x)
    x = Activation('tanh')(x)

    generator_unet = Model(input=[unet_input], outputs=[x])
    return generator_unet


def DCGAN_discriminator(img_shape, disc_img_shape, patch_num, model_name='DCGAN_discriminator'):
    disc_raw_img_shape = (disc_img_shape[0], disc_img_shape[1], img_shape[-1])
    list_input = [Input(shape=disc_img_shape, name='disc_input_'+str(i)) for i in range(patch_num)]
    list_raw_input = [Input(shape=disc_raw_img_shape, name='disc_raw_input_'+str(i)) for i in range(patch_num)]

    axis_num = -1
    filters_num = 64
    conv_num = int(np.floor(np.log(disc_img_shape[1])/np.log(2)))
    list_filters = [filters_num*min(8, (2**i)) for i in range(conv_num)]

    # First Conv
    generated_patch_input = Input(shape=disc_img_shape, name='discriminator_input')
    xg = Conv2D(list_filters[0], (3,3), strides=(2,2), name='disc_conv2d_1', padding='same')(generated_patch_input)
    xg = BatchNormalization(axis=axis_num)(xg)
    xg = LeakyReLU(0.2)(xg)

    # First Raw Conv
    raw_patch_input = Input(shape=disc_raw_img_shape, name='discriminator_raw_input')
    xr = Conv2D(list_filters[0], (3,3), strides=(2,2), name='raw_disc_conv2d_1', padding='same')(raw_patch_input)
    xr = BatchNormalization(axis=axis_num)(xr)
    xr = LeakyReLU(0.2)(xr)

    # Next Conv
    for i, f in enumerate(list_filters[1:]):
        name = 'disc_conv2d_' + str(i+2)
        x = Concatenate(axis=axis_num)([xg, xr])
        x = Conv2D(f, (3,3), strides=(2,2), name=name, padding='same')(x)
        x = BatchNormalization(axis=axis_num)(x)
        x = LeakyReLU(0.2)(x)

    x_flat = Flatten()(x)
    x = Dense(2, activation='softmax', name='disc_dense')(x_flat)

    PatchGAN = Model(inputs=[generated_patch_input, raw_patch_input], outputs=[x], name='PatchGAN')
    print('PatchGan summary')
    PatchGAN.summary()

    x = [PatchGAN([list_input[i], list_raw_input[i]]) for i in range(patch_num)]

    if len(x)>1:
        x = Concatenate(axis=axis_num)(x)
    else:
        x = x[0]

    x_out = Dense(2, activation='softmax', name='disc_output')(x)

    discriminator_model = Model(inputs=(list_input+list_raw_input), outputs=[x_out], name=model_name)
    

    return discriminator_model


def DCGAN(generator, discriminator, img_shape, patch_size):
    raw_input = Input(shape=img_shape, name='DCGAN_input')
    genarated_image = generator(raw_input)

    h, w = img_shape[:-1]
    ph, pw = patch_size, patch_size

    list_row_idx = [(i*ph, (i+1)*ph) for i in range(h//ph)]
    list_col_idx = [(i*pw, (i+1)*pw) for i in range(w//pw)]

    list_gen_patch = []
    list_raw_patch = []
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            raw_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(raw_input)
            list_raw_patch.append(raw_patch)
            x_patch = Lambda(lambda z: z[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])(genarated_image)
            list_gen_patch.append(x_patch)

    DCGAN_output = discriminator(list_gen_patch+list_raw_patch)
    
    DCGAN = Model(inputs=[raw_input],
                  outputs=[genarated_image, DCGAN_output],
                  name='DCGAN')

    return DCGAN



def my_load_generator(img_shape, disc_img_shape):
    model = generator_unet_upsampling(img_shape, disc_img_shape)
    model.summary()
    return model

def my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num):
    model = DCGAN_discriminator(img_shape, disc_img_shape, patch_num)
    model.summary()
    return model

def my_load_DCGAN(generator, discriminator, img_shape, patch_size):
    model = DCGAN(generator, discriminator, img_shape, patch_size)
    return model

"""
Good!
    for i, f in enumerate(list_filters_num[1:]):
        name = "unet_upconv2D_" + str(i+2)
        if i<2:
            d = True
            up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        else:
            d = False
            up_conv = up_conv_block_unet_alt(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        #up_conv = up_conv_block_unet(list_decoder[-1], list_encoder[-(i+3)], f, name, axis_num, dropout=d)
        list_decoder.append(up_conv)

generator_unet = Model(input=[unet_input], outputs=[x])
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
unet_input (InputLayer)          (None, 128, 128, 1)   0
____________________________________________________________________________________________________
unet_conv2D_1 (Conv2D)           (None, 64, 64, 64)    640         unet_input[0][0]
____________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 64, 64, 64)    0           unet_conv2D_1[0][0]
____________________________________________________________________________________________________
unet_conv2D_2 (Conv2D)           (None, 32, 32, 128)   73856       leaky_re_lu_1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 32, 32, 128)   512         unet_conv2D_2[0][0]
____________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 32, 32, 128)   0           batch_normalization_1[0][0]
____________________________________________________________________________________________________
unet_conv2D_3 (Conv2D)           (None, 16, 16, 256)   295168      leaky_re_lu_2[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 16, 16, 256)   1024        unet_conv2D_3[0][0]
____________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 16, 16, 256)   0           batch_normalization_2[0][0]
____________________________________________________________________________________________________
unet_conv2D_4 (Conv2D)           (None, 8, 8, 512)     1180160     leaky_re_lu_3[0][0]
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 8, 8, 512)     2048        unet_conv2D_4[0][0]
____________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 8, 8, 512)     0           batch_normalization_3[0][0]
____________________________________________________________________________________________________
unet_conv2D_5 (Conv2D)           (None, 4, 4, 512)     2359808     leaky_re_lu_4[0][0]
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 4, 4, 512)     2048        unet_conv2D_5[0][0]
____________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)        (None, 4, 4, 512)     0           batch_normalization_4[0][0]
____________________________________________________________________________________________________
unet_conv2D_6 (Conv2D)           (None, 2, 2, 512)     2359808     leaky_re_lu_5[0][0]
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 2, 2, 512)     2048        unet_conv2D_6[0][0]
____________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 2, 2, 512)     0           batch_normalization_5[0][0]
____________________________________________________________________________________________________
unet_conv2D_7 (Conv2D)           (None, 1, 1, 512)     2359808     leaky_re_lu_6[0][0]
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 1, 1, 512)     2048        unet_conv2D_7[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1, 1, 512)     0           batch_normalization_6[0][0]
____________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)   (None, 2, 2, 512)     0           activation_1[0][0]
____________________________________________________________________________________________________
unet_upconv2D_1 (Conv2D)         (None, 2, 2, 512)     2359808     up_sampling2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 2, 2, 512)     2048        unet_upconv2D_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2, 2, 512)     0           batch_normalization_7[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 2, 2, 1024)    0           dropout_1[0][0]
                                                                   batch_normalization_5[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 2, 2, 1024)    0           concatenate_1[0][0]
____________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)   (None, 4, 4, 1024)    0           activation_2[0][0]
____________________________________________________________________________________________________
unet_upconv2D_2 (Conv2D)         (None, 4, 4, 512)     4719104     up_sampling2d_2[0][0]
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 4, 4, 512)     2048        unet_upconv2D_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 4, 4, 512)     0           batch_normalization_8[0][0]
____________________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 4, 4, 1024)    0           dropout_2[0][0]
                                                                   batch_normalization_4[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 4, 4, 1024)    0           concatenate_2[0][0]
____________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)   (None, 8, 8, 1024)    0           activation_3[0][0]
____________________________________________________________________________________________________
unet_upconv2D_3 (Conv2D)         (None, 8, 8, 256)     2359552     up_sampling2d_3[0][0]
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 8, 8, 256)     1024        unet_upconv2D_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 8, 8, 256)     0           batch_normalization_9[0][0]
____________________________________________________________________________________________________
concatenate_3 (Concatenate)      (None, 8, 8, 768)     0           dropout_3[0][0]
                                                                   batch_normalization_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 8, 8, 768)     0           concatenate_3[0][0]
____________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)   (None, 16, 16, 768)   0           activation_4[0][0]
____________________________________________________________________________________________________
unet_upconv2D_4 (Conv2D)         (None, 16, 16, 128)   884864      up_sampling2d_4[0][0]
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 16, 16, 128)   512         unet_upconv2D_4[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 16, 16, 128)   0           batch_normalization_10[0][0]
____________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)   (None, 32, 32, 128)   0           activation_5[0][0]
____________________________________________________________________________________________________
unet_upconv2D_5 (Conv2D)         (None, 32, 32, 64)    73792       up_sampling2d_5[0][0]
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 32, 32, 64)    256         unet_upconv2D_5[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 32, 32, 64)    0           batch_normalization_11[0][0]
____________________________________________________________________________________________________
up_sampling2d_6 (UpSampling2D)   (None, 64, 64, 64)    0           activation_6[0][0]
____________________________________________________________________________________________________
unet_upconv2D_6 (Conv2D)         (None, 64, 64, 64)    36928       up_sampling2d_6[0][0]
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 64, 64, 64)    256         unet_upconv2D_6[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 64, 64, 64)    0           batch_normalization_12[0][0]
____________________________________________________________________________________________________
up_sampling2d_7 (UpSampling2D)   (None, 128, 128, 64)  0           activation_7[0][0]
____________________________________________________________________________________________________
last_conv (Conv2D)               (None, 128, 128, 3)   1731        up_sampling2d_7[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 128, 128, 3)   0           last_conv[0][0]
====================================================================================================
Total params: 19,080,899
Trainable params: 19,072,963
Non-trainable params: 7,936
____________________________________________________________________________________________________
PatchGan summary
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
discriminator_input (InputLayer) (None, 64, 64, 3)     0
____________________________________________________________________________________________________
discriminator_raw_input (InputLa (None, 64, 64, 1)     0
____________________________________________________________________________________________________
disc_conv2d_1 (Conv2D)           (None, 32, 32, 64)    1792        discriminator_input[0][0]
____________________________________________________________________________________________________
raw_disc_conv2d_1 (Conv2D)       (None, 32, 32, 64)    640         discriminator_raw_input[0][0]
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 32, 32, 64)    256         disc_conv2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 32, 32, 64)    256         raw_disc_conv2d_1[0][0]
____________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 32, 32, 64)    0           batch_normalization_13[0][0]
____________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 32, 32, 64)    0           batch_normalization_14[0][0]
____________________________________________________________________________________________________
concatenate_8 (Concatenate)      (None, 32, 32, 128)   0           leaky_re_lu_7[0][0]
                                                                   leaky_re_lu_8[0][0]
____________________________________________________________________________________________________
disc_conv2d_6 (Conv2D)           (None, 16, 16, 512)   590336      concatenate_8[0][0]
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 16, 16, 512)   2048        disc_conv2d_6[0][0]
____________________________________________________________________________________________________
leaky_re_lu_13 (LeakyReLU)       (None, 16, 16, 512)   0           batch_normalization_19[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 131072)        0           leaky_re_lu_13[0][0]
____________________________________________________________________________________________________
disc_dense (Dense)               (None, 2)             262146      flatten_1[0][0]
====================================================================================================
Total params: 857,474
Trainable params: 856,194
Non-trainable params: 1,280
____________________________________________________________________________________________________
  generator_unet = Model(input=[unet_input], outputs=[x])
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
unet_input (InputLayer)          (None, 128, 128, 1)   0
____________________________________________________________________________________________________
unet_conv2D_1 (Conv2D)           (None, 64, 64, 64)    640         unet_input[0][0]
____________________________________________________________________________________________________
leaky_re_lu_1 (LeakyReLU)        (None, 64, 64, 64)    0           unet_conv2D_1[0][0]
____________________________________________________________________________________________________
unet_conv2D_2 (Conv2D)           (None, 32, 32, 128)   73856       leaky_re_lu_1[0][0]
____________________________________________________________________________________________________
batch_normalization_1 (BatchNorm (None, 32, 32, 128)   512         unet_conv2D_2[0][0]
____________________________________________________________________________________________________
leaky_re_lu_2 (LeakyReLU)        (None, 32, 32, 128)   0           batch_normalization_1[0][0]
____________________________________________________________________________________________________
unet_conv2D_3 (Conv2D)           (None, 16, 16, 256)   295168      leaky_re_lu_2[0][0]
____________________________________________________________________________________________________
batch_normalization_2 (BatchNorm (None, 16, 16, 256)   1024        unet_conv2D_3[0][0]
____________________________________________________________________________________________________
leaky_re_lu_3 (LeakyReLU)        (None, 16, 16, 256)   0           batch_normalization_2[0][0]
____________________________________________________________________________________________________
unet_conv2D_4 (Conv2D)           (None, 8, 8, 512)     1180160     leaky_re_lu_3[0][0]
____________________________________________________________________________________________________
batch_normalization_3 (BatchNorm (None, 8, 8, 512)     2048        unet_conv2D_4[0][0]
____________________________________________________________________________________________________
leaky_re_lu_4 (LeakyReLU)        (None, 8, 8, 512)     0           batch_normalization_3[0][0]
____________________________________________________________________________________________________
unet_conv2D_5 (Conv2D)           (None, 4, 4, 512)     2359808     leaky_re_lu_4[0][0]
____________________________________________________________________________________________________
batch_normalization_4 (BatchNorm (None, 4, 4, 512)     2048        unet_conv2D_5[0][0]
____________________________________________________________________________________________________
leaky_re_lu_5 (LeakyReLU)        (None, 4, 4, 512)     0           batch_normalization_4[0][0]
____________________________________________________________________________________________________
unet_conv2D_6 (Conv2D)           (None, 2, 2, 512)     2359808     leaky_re_lu_5[0][0]
____________________________________________________________________________________________________
batch_normalization_5 (BatchNorm (None, 2, 2, 512)     2048        unet_conv2D_6[0][0]
____________________________________________________________________________________________________
leaky_re_lu_6 (LeakyReLU)        (None, 2, 2, 512)     0           batch_normalization_5[0][0]
____________________________________________________________________________________________________
unet_conv2D_7 (Conv2D)           (None, 1, 1, 512)     2359808     leaky_re_lu_6[0][0]
____________________________________________________________________________________________________
batch_normalization_6 (BatchNorm (None, 1, 1, 512)     2048        unet_conv2D_7[0][0]
____________________________________________________________________________________________________
activation_1 (Activation)        (None, 1, 1, 512)     0           batch_normalization_6[0][0]
____________________________________________________________________________________________________
up_sampling2d_1 (UpSampling2D)   (None, 2, 2, 512)     0           activation_1[0][0]
____________________________________________________________________________________________________
unet_upconv2D_1 (Conv2D)         (None, 2, 2, 512)     2359808     up_sampling2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_7 (BatchNorm (None, 2, 2, 512)     2048        unet_upconv2D_1[0][0]
____________________________________________________________________________________________________
dropout_1 (Dropout)              (None, 2, 2, 512)     0           batch_normalization_7[0][0]
____________________________________________________________________________________________________
concatenate_1 (Concatenate)      (None, 2, 2, 1024)    0           dropout_1[0][0]
                                                                   batch_normalization_5[0][0]
____________________________________________________________________________________________________
activation_2 (Activation)        (None, 2, 2, 1024)    0           concatenate_1[0][0]
____________________________________________________________________________________________________
up_sampling2d_2 (UpSampling2D)   (None, 4, 4, 1024)    0           activation_2[0][0]
____________________________________________________________________________________________________
unet_upconv2D_2 (Conv2D)         (None, 4, 4, 512)     4719104     up_sampling2d_2[0][0]
____________________________________________________________________________________________________
batch_normalization_8 (BatchNorm (None, 4, 4, 512)     2048        unet_upconv2D_2[0][0]
____________________________________________________________________________________________________
dropout_2 (Dropout)              (None, 4, 4, 512)     0           batch_normalization_8[0][0]
____________________________________________________________________________________________________
concatenate_2 (Concatenate)      (None, 4, 4, 1024)    0           dropout_2[0][0]
                                                                   batch_normalization_4[0][0]
____________________________________________________________________________________________________
activation_3 (Activation)        (None, 4, 4, 1024)    0           concatenate_2[0][0]
____________________________________________________________________________________________________
up_sampling2d_3 (UpSampling2D)   (None, 8, 8, 1024)    0           activation_3[0][0]
____________________________________________________________________________________________________
unet_upconv2D_3 (Conv2D)         (None, 8, 8, 256)     2359552     up_sampling2d_3[0][0]
____________________________________________________________________________________________________
batch_normalization_9 (BatchNorm (None, 8, 8, 256)     1024        unet_upconv2D_3[0][0]
____________________________________________________________________________________________________
dropout_3 (Dropout)              (None, 8, 8, 256)     0           batch_normalization_9[0][0]
____________________________________________________________________________________________________
concatenate_3 (Concatenate)      (None, 8, 8, 768)     0           dropout_3[0][0]
                                                                   batch_normalization_3[0][0]
____________________________________________________________________________________________________
activation_4 (Activation)        (None, 8, 8, 768)     0           concatenate_3[0][0]
____________________________________________________________________________________________________
up_sampling2d_4 (UpSampling2D)   (None, 16, 16, 768)   0           activation_4[0][0]
____________________________________________________________________________________________________
unet_upconv2D_4 (Conv2D)         (None, 16, 16, 128)   884864      up_sampling2d_4[0][0]
____________________________________________________________________________________________________
batch_normalization_10 (BatchNor (None, 16, 16, 128)   512         unet_upconv2D_4[0][0]
____________________________________________________________________________________________________
concatenate_4 (Concatenate)      (None, 16, 16, 384)   0           batch_normalization_10[0][0]
                                                                   batch_normalization_2[0][0]
____________________________________________________________________________________________________
activation_5 (Activation)        (None, 16, 16, 384)   0           concatenate_4[0][0]
____________________________________________________________________________________________________
up_sampling2d_5 (UpSampling2D)   (None, 32, 32, 384)   0           activation_5[0][0]
____________________________________________________________________________________________________
unet_upconv2D_5 (Conv2D)         (None, 32, 32, 64)    221248      up_sampling2d_5[0][0]
____________________________________________________________________________________________________
batch_normalization_11 (BatchNor (None, 32, 32, 64)    256         unet_upconv2D_5[0][0]
____________________________________________________________________________________________________
concatenate_5 (Concatenate)      (None, 32, 32, 192)   0           batch_normalization_11[0][0]
                                                                   batch_normalization_1[0][0]
____________________________________________________________________________________________________
activation_6 (Activation)        (None, 32, 32, 192)   0           concatenate_5[0][0]
____________________________________________________________________________________________________
up_sampling2d_6 (UpSampling2D)   (None, 64, 64, 192)   0           activation_6[0][0]
____________________________________________________________________________________________________
unet_upconv2D_6 (Conv2D)         (None, 64, 64, 64)    110656      up_sampling2d_6[0][0]
____________________________________________________________________________________________________
batch_normalization_12 (BatchNor (None, 64, 64, 64)    256         unet_upconv2D_6[0][0]
____________________________________________________________________________________________________
concatenate_6 (Concatenate)      (None, 64, 64, 128)   0           batch_normalization_12[0][0]
                                                                   unet_conv2D_1[0][0]
____________________________________________________________________________________________________
activation_7 (Activation)        (None, 64, 64, 128)   0           concatenate_6[0][0]
____________________________________________________________________________________________________
up_sampling2d_7 (UpSampling2D)   (None, 128, 128, 128) 0           activation_7[0][0]
____________________________________________________________________________________________________
last_conv (Conv2D)               (None, 128, 128, 3)   3459        up_sampling2d_7[0][0]
____________________________________________________________________________________________________
activation_8 (Activation)        (None, 128, 128, 3)   0           last_conv[0][0]
====================================================================================================
Total params: 19,303,811
Trainable params: 19,295,875
Non-trainable params: 7,936
____________________________________________________________________________________________________
PatchGan summary
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
discriminator_input (InputLayer) (None, 64, 64, 3)     0
____________________________________________________________________________________________________
discriminator_raw_input (InputLa (None, 64, 64, 1)     0
____________________________________________________________________________________________________
disc_conv2d_1 (Conv2D)           (None, 32, 32, 64)    1792        discriminator_input[0][0]
____________________________________________________________________________________________________
raw_disc_conv2d_1 (Conv2D)       (None, 32, 32, 64)    640         discriminator_raw_input[0][0]
____________________________________________________________________________________________________
batch_normalization_13 (BatchNor (None, 32, 32, 64)    256         disc_conv2d_1[0][0]
____________________________________________________________________________________________________
batch_normalization_14 (BatchNor (None, 32, 32, 64)    256         raw_disc_conv2d_1[0][0]
____________________________________________________________________________________________________
leaky_re_lu_7 (LeakyReLU)        (None, 32, 32, 64)    0           batch_normalization_13[0][0]
____________________________________________________________________________________________________
leaky_re_lu_8 (LeakyReLU)        (None, 32, 32, 64)    0           batch_normalization_14[0][0]
____________________________________________________________________________________________________
concatenate_11 (Concatenate)     (None, 32, 32, 128)   0           leaky_re_lu_7[0][0]
                                                                   leaky_re_lu_8[0][0]
____________________________________________________________________________________________________
disc_conv2d_6 (Conv2D)           (None, 16, 16, 512)   590336      concatenate_11[0][0]
____________________________________________________________________________________________________
batch_normalization_19 (BatchNor (None, 16, 16, 512)   2048        disc_conv2d_6[0][0]
____________________________________________________________________________________________________
leaky_re_lu_13 (LeakyReLU)       (None, 16, 16, 512)   0           batch_normalization_19[0][0]
____________________________________________________________________________________________________
flatten_1 (Flatten)              (None, 131072)        0           leaky_re_lu_13[0][0]
____________________________________________________________________________________________________
disc_dense (Dense)               (None, 2)             262146      flatten_1[0][0]
====================================================================================================
Total params: 857,474
Trainable params: 856,194
Non-trainable params: 1,280
____________________________________________________________________________________________________
____________________________________________________________________________________________________
Layer (type)                     Output Shape          Param #     Connected to
====================================================================================================
disc_input_0 (InputLayer)        (None, 64, 64, 3)     0
____________________________________________________________________________________________________
disc_raw_input_0 (InputLayer)    (None, 64, 64, 1)     0
____________________________________________________________________________________________________
disc_input_1 (InputLayer)        (None, 64, 64, 3)     0
____________________________________________________________________________________________________
disc_raw_input_1 (InputLayer)    (None, 64, 64, 1)     0
____________________________________________________________________________________________________
disc_input_2 (InputLayer)        (None, 64, 64, 3)     0
____________________________________________________________________________________________________
disc_raw_input_2 (InputLayer)    (None, 64, 64, 1)     0
____________________________________________________________________________________________________
disc_input_3 (InputLayer)        (None, 64, 64, 3)     0
____________________________________________________________________________________________________
disc_raw_input_3 (InputLayer)    (None, 64, 64, 1)     0
____________________________________________________________________________________________________
PatchGAN (Model)                 (None, 2)             857474      disc_input_0[0][0]
                                                                   disc_raw_input_0[0][0]
                                                                   disc_input_1[0][0]
                                                                   disc_raw_input_1[0][0]
                                                                   disc_input_2[0][0]
                                                                   disc_raw_input_2[0][0]
                                                                   disc_input_3[0][0]
                                                                   disc_raw_input_3[0][0]
____________________________________________________________________________________________________
concatenate_12 (Concatenate)     (None, 8)             0           PatchGAN[1][0]
                                                                   PatchGAN[2][0]
                                                                   PatchGAN[3][0]
                                                                   PatchGAN[4][0]
____________________________________________________________________________________________________
disc_output (Dense)              (None, 2)             18          concatenate_12[0][0]
====================================================================================================
Total params: 857,492
Trainable params: 856,212
Non-trainable params: 1,280
____________________________________________________________________________________________________
"""
