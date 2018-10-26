
# Transfer Learning Example

This demo shows how to train a ConvNet on your own data using pretrained weights.
In the example we will train an image classifier to recognize food.
The demo is implemented using TensorFlow 1.10

## Main components

1. Model definition
    * In the demo we will use VGG19 model
    * The model is defined in `model.py` using TensorFlow Slim
2. Training pipeline
    * The pipeline contains train/val dataset pipelines, train/val graphs definition, train loss, validation accuracy, loading of pretrained weights from a snapshot, various TF training structures and training loop
    * The procedure is defined in `train.py`
    * To run the training use `train.py` or `run_train.sh`
    * Check "Settings" section in both files before running
3. Inference pipeline
    * The pipeline contains inference graph definition, loading trained weights, loading and preprocessing an input test image and inference procedure
    * To run the inference use `inference.py`
    * Check "Settings" section before running
4. Pretrained weights
    * For VGG19 we can download pretrained weights from here: https://github.com/tensorflow/models/tree/master/research/slim
    * The snapshot should be located at `data/vgg_19.ckpt`
5. Dataset
    * For the demo we will use Food-101 dataset: https://www.vision.ee.ethz.ch/datasets_extra/food-101/
    * The dataset should be located at `data/food-101/`

## Docker

To avoid problems with various versions of the frameworks, it is recommended to execute everything in a docker container.

* To build the docker container execute `./docker/docker_build.sh`
* To run the docker container in bash mode execute `./docker/docker_run_bash.sh`
