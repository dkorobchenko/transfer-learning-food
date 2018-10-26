#!/bin/bash

################## Settings ##################
GPU_ID=0
OUTPUT_DIR="checkpoints/vgg19_food"
##############################################

export CUDA_VISIBLE_DEVICES=$GPU_ID

mkdir -p $OUTPUT_DIR
cp train.py $OUTPUT_DIR

CMD="python train.py"
$CMD 2>&1 | tee -a $OUTPUT_DIR"/log.txt"
