'''
TF definition of VGG19 model
PDF: http://arxiv.org/pdf/1409.1556.pdf

Author: Dmitry Korobchenko (dkorobchenko@nvidia.com)
'''

import tensorflow as tf
import tensorflow.contrib.slim as slim

def vgg_19(inputs,
           num_classes,
           is_training,
           scope='vgg_19',
           weight_decay=0.0005):
    with slim.arg_scope([slim.conv2d],
                activation_fn=tf.nn.relu,
                weights_regularizer=slim.l2_regularizer(weight_decay),
                biases_initializer=tf.zeros_initializer(),
                padding='SAME'):
        with tf.variable_scope(scope, 'vgg_19', [inputs]):
            net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 4, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 4, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            # Use conv2d instead of fully_connected layers
            net = slim.conv2d(net, 4096, [7, 7], padding='VALID', scope='fc6')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='drop6')
            net = slim.conv2d(net, 4096, [1, 1], scope='fc7')
            net = slim.dropout(net, 0.5, is_training=is_training, scope='drop7')
            net = slim.conv2d(net, num_classes, [1, 1], scope='fc8',
                activation_fn=None)
            net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
    return net
