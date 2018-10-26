'''
Train image classifier on Food-101 dataset via VGG19 fine-tuning

Author: Dmitry Korobchenko (dkorobchenko@nvidia.com)
'''

import numpy as np

import tensorflow as tf
import tensorflow.contrib.slim as slim
tf.logging.set_verbosity(tf.logging.INFO)

import model
import data

###########################################################
###  Settings
###########################################################

DATASET_ROOT = 'data/food-101/'
INPUT_SIZE = 224
RANDOM_CROP_MARGIN = 10
TRAIN_EPOCHS = 20
TRAIN_BATCH_SIZE = 64
VAL_BATCH_SIZE = 128
LR_START = 0.001
LR_END = LR_START / 1e4
MOMENTUM = 0.9
VGG_PRETRAINED_CKPT = 'data/vgg_19.ckpt'
CHECKPOINT_DIR = 'checkpoints/vgg19_food'
LOG_LOSS_EVERY = 10
CALC_ACC_EVERY = 500

###########################################################
###  Get Food-101 dataset as train/validation lists
###  of img file paths and labels
###########################################################

train_data, val_data, classes = data.food101(DATASET_ROOT)
num_classes = len(classes)

###########################################################
###  Build training and validation data pipelines
###########################################################

train_ds, train_iters = data.train_dataset(train_data,
    TRAIN_BATCH_SIZE, TRAIN_EPOCHS, INPUT_SIZE, RANDOM_CROP_MARGIN)
train_ds_iterator = train_ds.make_one_shot_iterator()
train_x, train_y = train_ds_iterator.get_next()

val_ds, val_iters = data.val_dataset(val_data,
    VAL_BATCH_SIZE, INPUT_SIZE)
val_ds_iterator = val_ds.make_initializable_iterator()
val_x, val_y = val_ds_iterator.get_next()

###########################################################
###  Construct training and validation graphs
###########################################################

with tf.variable_scope('', reuse=tf.AUTO_REUSE):
    train_logits = model.vgg_19(train_x, num_classes, is_training=True)
    val_logits = model.vgg_19(val_x, num_classes, is_training=False)

###########################################################
###  Construct training loss
###########################################################

loss = tf.losses.sparse_softmax_cross_entropy(
    labels=train_y, logits=train_logits)
tf.summary.scalar('loss', loss)

###########################################################
###  Construct validation accuracy
###  and related functions
###########################################################

def calc_accuracy(sess, val_logits, val_y, val_iters):
    acc_total = 0.0
    acc_denom = 0
    for i in range(val_iters):
        logits, y = sess.run((val_logits, val_y))
        y_pred = np.argmax(logits, axis=1)
        correct = np.count_nonzero(y == y_pred)
        acc_denom += y_pred.shape[0]
        acc_total += float(correct)
        tf.logging.info('Validating batch [{} / {}] correct = {}'.format(
            i, val_iters, correct))
    acc_total /= acc_denom
    return acc_total

def accuracy_summary(sess, acc_value, iteration):
    acc_summary = tf.Summary()
    acc_summary.value.add(tag="accuracy", simple_value=acc_value)
    sess._hooks[1]._summary_writer.add_summary(acc_summary, iteration)

###########################################################
###  Define set of VGG variables to restore
###  Create the Restorer
###  Define init callback (used by monitored session)
###########################################################

vars_to_restore = tf.contrib.framework.get_variables_to_restore(
    exclude=['vgg_19/fc8'])
vgg_restorer = tf.train.Saver(vars_to_restore)

def init_fn(scaffold, sess):
    vgg_restorer.restore(sess, VGG_PRETRAINED_CKPT)

###########################################################
###  Create various training structures
###########################################################

global_step = tf.train.get_or_create_global_step()
lr = tf.train.polynomial_decay(LR_START, global_step, train_iters, LR_END)
tf.summary.scalar('learning_rate', lr)
optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=MOMENTUM)
training_op = slim.learning.create_train_op(
    loss, optimizer, global_step=global_step)
scaffold = tf.train.Scaffold(init_fn=init_fn)

###########################################################
###  Create monitored session
###  Run training loop
###########################################################

with tf.train.MonitoredTrainingSession(checkpoint_dir=CHECKPOINT_DIR,
                                       save_checkpoint_secs=600,
                                       save_summaries_steps=30,
                                       scaffold=scaffold) as sess:
    start_iter = sess.run(global_step)
    for iteration in range(start_iter, train_iters):

        # Gradient Descent
        loss_value = sess.run(training_op)

        # Loss logging
        if iteration % LOG_LOSS_EVERY == 0:
            tf.logging.info('[{} / {}] Loss = {}'.format(
                iteration, train_iters, loss_value))

        # Accuracy logging
        if iteration % CALC_ACC_EVERY == 0:
            sess.run(val_ds_iterator.initializer)
            acc_value = calc_accuracy(sess, val_logits, val_y, val_iters)
            accuracy_summary(sess, acc_value, iteration)
            tf.logging.info('[{} / {}] Validation accuracy = {}'.format(
                iteration, train_iters, acc_value))
