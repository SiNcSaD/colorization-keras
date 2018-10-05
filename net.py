import keras
import tensorflow as tf
import numpy as np
import os
nilboy_weight_decay = 0.001

class Net():
    def __init__(self, train=True, common_params=None, net_params=None):
        self.train = train
        self.weight_decay = 0.0

    def create_model(self, input_shape):
        input_image = keras.layers.Input(batch_shape=input_shape, name='input')

        # conv1
        temp_conv = conv2d('bw_conv1_1', input_image, 64, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv1_2', temp_conv, 64, (3, 3), stride=2, wd=self.weight_decay)

        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv1_2norm')(temp_conv)

        # conv2
        temp_conv = conv2d('conv2_1', temp_conv, 128, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv2_2', temp_conv, 128, (3, 3), stride=2, wd=self.weight_decay)

        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv2_2norm')(temp_conv)

        # conv3
        temp_conv = conv2d('conv3_1', temp_conv, 256, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv3_2', temp_conv, 256, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv3_3', temp_conv, 256, (3, 3), stride=2, wd=self.weight_decay)

        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv3_3norm')(temp_conv)

        # conv4
        temp_conv = conv2d('conv4_1', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv4_2', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv4_3', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)
        
        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv4_3norm')(temp_conv)

        # conv5
        temp_conv = conv2d('conv5_1', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)

        temp_conv = conv2d('conv5_2', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)

        temp_conv = conv2d('conv5_3', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)
        
        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv5_3norm')(temp_conv)

        # conv6
        temp_conv = conv2d('conv6_1', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)

        temp_conv = conv2d('conv6_2', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)

        temp_conv = conv2d('conv6_3', temp_conv, 512, (3, 3), stride=1, dilation=2, wd=self.weight_decay)
        
        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv6_3norm')(temp_conv)

        # conv7
        temp_conv = conv2d('conv7_1', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv7_2', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv7_3', temp_conv, 512, (3, 3), stride=1, wd=self.weight_decay)
        
        temp_conv = keras.layers.BatchNormalization(center=False, scale=False, trainable=False, epsilon=1e-5, name='conv7_3norm')(temp_conv)

        # conv8
        temp_conv = deconv2d('conv8_1', temp_conv, 256, (4, 4), stride=2)

        temp_conv = conv2d('conv8_2', temp_conv, 256, (3, 3), stride=1, wd=self.weight_decay)

        temp_conv = conv2d('conv8_3', temp_conv, 256, (3, 3), stride=1, wd=self.weight_decay)
        
        conv8_313 = conv2d('conv8_313', temp_conv, 313, (1, 1), stride=1, relu=False, wd=self.weight_decay)

        model = keras.models.Model(input_image, conv8_313)

        return model

def conv2d(name, x, filters, kernel_size, pad='SAME', stride=1, wd=nilboy_weight_decay, dilation=1, relu=True):
    if stride == 2:
        x = keras.layers.ZeroPadding2D()(x)

    conv = keras.layers.Conv2D(filters,
                               kernel_size,
                               strides=(stride, stride),
                               padding=pad,
                               dilation_rate=(dilation, dilation),
                               name=name)(x)

    if stride == 2:
        # h, w, c = [l.value for l in conv.shape[1:]]
        # conv = keras.backend.slice(conv, [0, 0, 0, 0], [1, h-1, w-1, c])
        conv = keras.layers.Cropping2D(cropping=((0, 1), (0, 1)))(conv)
        # conv = crop()(conv)
    
    if relu:
        conv1 = keras.layers.Activation('relu')(conv)
    else:
        conv1 = conv

    return conv1

def deconv2d(name, x, filters, kernel_size, stride=1):
    deconv = keras.layers.Conv2DTranspose(filters, 
                                          kernel_size,
                                          strides=(stride, stride),
                                          padding='SAME',
                                          dilation_rate=(1, 1),
                                          name=name)(x)
    deconv1 = keras.layers.Activation('relu')(deconv)
    return deconv1

def crop():
    # Crops (or slices) a Tensor on a given dimension from start to end
    # example : to crop tensor x[:, :, 5:10]
    # call slice(2, 5, 10) as you want to crop on the second dimension
    def func(x):
        return x[:, :-1, :-1, :]
    return keras.layers.Lambda(func)