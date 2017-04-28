#! /bin/sh
python -i resnet_imagenet.py --model=imagenet1k-resnet-152 --data-val=./data/imagenet/imagenet1k-val.rec --gpus=0 --data-nthreads=60
