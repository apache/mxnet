#!/bin/bash

pip install gluoncv

python ${MXNET_HOME}/tests/python/tensorrt/test_tensorrt_resnet_resnext.py
