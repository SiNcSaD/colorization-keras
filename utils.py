import os
import numpy as np
from skimage import color
from skimage.transform import resize
import scipy.ndimage.interpolation as sni

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.expand_dims(np.max(x, axis=-1), axis=-1))
    return e_x / np.expand_dims(e_x.sum(axis=-1), axis=-1) # only difference

def decode(data_l, conv8_313, rebalance=1):
  """
  Args:
    data_l   : [1, height, width, 1]
    conv8_313: [1, height/4, width/4, 313]
  Returns:
    img_rgb  : [height, width, 3]
  """
  height, width = data_l.shape
  conv8_313 = conv8_313[0, :, :, :]
  conv8_313_rh = conv8_313 * rebalance
  class8_313_rh = softmax(conv8_313_rh)

  cc = np.load('pts_in_hull.npy')
  
  data_ab = np.dot(class8_313_rh, cc)    # (212, 318, 2)
  # data_ab_us = resize(data_ab, (height, width))    # (846, 1269, 2)
  data_ab_us = sni.zoom(data_ab,(1.*height/56,1.*width/56,1))
  img_lab = np.concatenate((data_l[:,:,np.newaxis], data_ab_us), axis=-1)    # (846, 1269, 3)
  img_rgb = (255*np.clip(color.lab2rgb(img_lab),0,1)).astype('uint8')
  # img_rgb = (255*color.lab2rgb(img_lab)).astype('uint8')

  return img_rgb