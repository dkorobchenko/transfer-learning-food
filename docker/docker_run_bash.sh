#!/bin/bash

nvidia-docker run \
    --rm -it \
    -v `pwd`:/transfer-learning/ \
    transfer-learning-tf:latest \
    /bin/bash
