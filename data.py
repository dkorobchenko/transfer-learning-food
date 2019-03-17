'''
Functions for parsing a dataset (e.g. Food-101)
and constructing train/val data pipelines
Food-101 source: https://www.vision.ee.ethz.ch/datasets_extra/food-101/

Author: Dmitry Korobchenko (dkorobchenko@nvidia.com)
'''

from os.path import join as opj

import tensorflow as tf

def parse_ds_subset(img_root, list_fpath, classes):
    '''
    Parse a meta file with image paths and labels
    -> img_root: path to the root of image folders
    -> list_fpath: path to the file with the list (e.g. train.txt)
    -> classes: list of class names
    <- (list_of_img_paths, integer_labels)
    '''
    fpaths = []
    labels = []

    with open(list_fpath, 'r') as f:
        for line in f:
            class_name, image_id = line.strip().split('/')
            fpaths.append(opj(img_root, class_name, image_id+'.jpg'))
            labels.append(classes.index(class_name))

    return fpaths, labels

def food101(dataset_root):
    '''
    Get lists of train and validation examples for Food-101 dataset
    -> dataset_root: root of the Food-101 dataset
    <- ((train_fpaths, train_labels), (val_fpaths, val_labels), classes)
    '''
    img_root = opj(dataset_root, 'images')
    train_list_fpath = opj(dataset_root, 'meta', 'train.txt')
    test_list_fpath = opj(dataset_root, 'meta', 'test.txt')
    classes_list_fpath = opj(dataset_root, 'meta', 'classes.txt')

    with open(classes_list_fpath, 'r') as f:
        classes = [line.strip() for line in f]

    train_data = parse_ds_subset(img_root, train_list_fpath, classes)
    val_data = parse_ds_subset(img_root, test_list_fpath, classes)

    return train_data, val_data, classes

def imread_and_crop(fpath, inp_size, margin=0, random_crop=False):
    '''
    Construct TF graph for image preparation:
    Read the file, crop and resize
    -> fpath: path to the JPEG image file (TF node)
    -> inp_size: size of the network input (e.g. 224)
    -> margin: cropping margin
    -> random_crop: perform random crop or central crop
    <- prepared image (TF node)
    '''
    data = tf.read_file(fpath)
    img = tf.image.decode_jpeg(data, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    shape = tf.shape(img)
    crop_size = tf.minimum(shape[0], shape[1]) - 2 * margin
    if random_crop:
        img = tf.random_crop(img, (crop_size, crop_size, 3))
    else: # central crop
        ho = (shape[0] - crop_size) // 2
        wo = (shape[1] - crop_size) // 2
        img = img[ho:ho+crop_size, wo:wo+crop_size, :]

    img = tf.image.resize_images(img, (inp_size, inp_size),
        method=tf.image.ResizeMethod.AREA)

    return img

def train_dataset(data, batch_size, epochs, inp_size, margin):
    '''
    Prepare training data pipeline
    -> data: (list_of_img_paths, integer_labels)
    -> batch_size: training batch size
    -> epochs: number of training epochs
    -> inp_size: size of the network input (e.g. 224)
    -> margin: cropping margin
    <- (dataset, number_of_train_iterations)
    '''
    num_examples = len(data[0])
    iters = (epochs * num_examples) // batch_size

    def fpath_to_image(fpath, label):
        img = imread_and_crop(fpath, inp_size, margin, random_crop=True)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.shuffle(buffer_size=num_examples)
    dataset = dataset.map(fpath_to_image)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, iters

def val_dataset(data, batch_size, inp_size):
    '''
    Prepare validation data pipeline
    -> data: (list_of_img_paths, integer_labels)
    -> batch_size: validation batch size
    -> inp_size: size of the network input (e.g. 224)
    <- (dataset, number_of_val_iterations)
    '''
    num_examples = len(data[0])
    iters = num_examples // batch_size

    def fpath_to_image(fpath, label):
        img = imread_and_crop(fpath, inp_size, 0, random_crop=False)
        return img, label

    dataset = tf.data.Dataset.from_tensor_slices(data)
    dataset = dataset.map(fpath_to_image)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    return dataset, iters
