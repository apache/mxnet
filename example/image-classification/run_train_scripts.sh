#!/bin/bash


#printf "[inception-resnet-v2]\n"
#python train_ccs-train.py \
#--network inception-resnet-v2 \
#--num-layers 50 \
#--model-prefix ./models/inception-resnet-v2-50-standford-512-lr-0.01

#printf "[resnext]\n"
#python train_ccs-train.py \
#--network resnext \
#--num-layers 50 \
#--model-prefix ./models/resnext-50-standford-512-lr-0.01

#printf "[resnet-50]\n"
#python train_ccs-train.py \
#--network resnet \
#--num-layers 50 \
#--model-prefix ./models/resnet-50-standford-512-lr-0.01

printf "[resnet-34]\n"
python train_ccs-train.py \
--network resnet \
--num-layers 34 \
--model-prefix ./models/resnet-34-standford-512-lr-0.01

