# ParaX

## Overview
This repository is for temporarily placing the ParaX project before publication. 
ParaX is a variation of MXNet 
that issues one instance processing a batch of samples for every core on a CPU, 
so as to alleviate memory bandwidth contention in DNN model training and inference on many-core CPUs.

Note: The key components of ParaX are temporarily not available 
until the paper is accepted, 
for hiding author info so as to satisfy the anonymity requirement.

## Install ParaX
Download the MXNet-1.5.0 source code from "https://github.com/apache/incubator-mxnet/".

Patch the diff file in the root directory of MXNet-1.5.0.

## Run ParaX
export MXNET_ENGINE_TYPE=NaiveEngine; 

python $MXNET_ROOT/tools/launch.py -n $number_of_instances -p $core_per_instance --launcher=local $command

$number_of_instances is the number of issued instances

$core_per_instance is the number of cores occupied by each instance

$command is the python program for DNN training or inference

When $number_of_instances is equal to the number of cores and using $core_per_instance=1, ParaX adopts the instance-per-core paradigm

## Example
export MXNET_ENGINE_TYPE=NaiveEngine; 

python $MXNET_ROOT /tools/launch.py -n 56 -p 1 --launcher=local python $MXNET_ROOT/example/image_classification/train_imagenet.py 

This will train ImageNet with ParaX.
