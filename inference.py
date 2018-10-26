'''
Perform image classification with a trained TF model

Author: Dmitry Korobchenko (dkorobchenko@nvidia.com)
'''

import sys
import numpy as np
import imageio
from skimage.transform import resize

import tensorflow as tf

import model

###########################################################
###  Settings
###########################################################

CLASSES_FPATH = 'data/food-101/meta/labels.txt'
INP_SIZE = 224 # Input will be cropped and resized
CHECKPOINT_DIR = 'checkpoints/vgg19_food'
IMG_FPATH = 'data/food-101/images/bruschetta/3564471.jpg'

###########################################################
###  Get all class names
###########################################################

with open(CLASSES_FPATH, 'r') as f:
    classes = [line.strip() for line in f]
num_classes = len(classes)

###########################################################
###  Construct inference graph
###########################################################

x = tf.placeholder(tf.float32, (1, INP_SIZE, INP_SIZE, 3), name='inputs')
logits = model.vgg_19(x, num_classes, is_training=False)

###########################################################
###  Create TF session and restore from a snapshot
###########################################################

sess = tf.Session()
snapshot_fpath = tf.train.latest_checkpoint(CHECKPOINT_DIR)
restorer = tf.train.Saver()
restorer.restore(sess, snapshot_fpath)

###########################################################
###  Load and prepare input image
###########################################################

def crop_and_resize(img, input_size):
    crop_size = min(img.shape[0], img.shape[1])
    ho = (img.shape[0] - crop_size) // 2
    wo = (img.shape[0] - crop_size) // 2
    img = img[ho:ho+crop_size, wo:wo+crop_size, :]
    img = resize(img, (input_size, input_size),
        order=3, mode='reflect', anti_aliasing=True, preserve_range=True)
    return img

img = imageio.imread(IMG_FPATH)
img = img.astype(np.float32)
img = crop_and_resize(img, INP_SIZE)
img = img[None, ...]

###########################################################
###  Run inference
###########################################################

out = sess.run(logits, feed_dict={x:img})
pred_class = classes[np.argmax(out)]

print('Input: {}'.format(IMG_FPATH))
print('Prediction: {}'.format(pred_class))
