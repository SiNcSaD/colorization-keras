import keras
import numpy as np
import os
import cv2
import pickle
from skimage import color
from net import Net
from utils import *
import tensorflowjs as tfjs

img = cv2.imread('./demo/ILSVRC2012_val_00046524.JPEG')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_rgb = (img_rgb.astype(dtype=np.float32)) / 255.0
img_rs = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_NEAREST)
img_lab_rs = color.rgb2lab(img_rs)
img_l_rs = img_lab_rs[:,:,0] - 50
img_l_rs = img_l_rs[None, :, :, None]
autocolor = Net(train=False)

model = autocolor.create_model([1, 224, 224, 1])

file_name = '{}.pkl'.format('caffe_weights')
with open(file_name, 'rb') as f:
    layer_dict = pickle.load(f)

layer_name_dict = dict([(layer.name, i) for i, layer in enumerate(model.layers)])

for layer in layer_dict:
    if layer['type'] != 'Convolution' and layer['type'] != 'Deconvolution':
        continue
    if layer['name'] == 'class8_ab':
        continue
    layer_idx = layer_name_dict[layer['name']]
    w = layer['weights'][0].transpose((2, 3, 1, 0))
    b = layer['weights'][1]
    model.layers[layer_idx].set_weights([w, b])

print('set weights finish')

for layer in layer_dict:
    if layer['type'] != 'BatchNorm':
        continue
    layer_idx = layer_name_dict[layer['name']]
    moving_mean = layer['weights'][0] / layer['weights'][2]
    moving_variance = layer['weights'][1] / layer['weights'][2]
    model.layers[layer_idx].set_weights([moving_mean, moving_variance])

print('set BN finish')

# model.save('color_model.h5')
# tfjs.converters.save_keras_model(model, 'web_tfjs_model')

print('Predicting...')
conv8_313 = model.predict(img_l_rs)

img_lab = color.rgb2lab(img_rgb)
img_l = img_lab[:,:,0]
img_rgb_out = decode(img_l, conv8_313, 2.606)
print('Saving...')
cv2.imwrite('color.jpg', cv2.cvtColor(img_rgb_out, cv2.COLOR_RGB2BGR))