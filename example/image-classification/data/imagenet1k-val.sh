#!/bin/bash

# This file download the imagnet-1k validation dataset and convert it into a rec
# file. One need to provide the URL for the ILSVRC2012_img_val.tar, which can be
# find at http://www.image-net.org/download-images
#
# Example usage (replace the URL with the correct one):
# ./imagenet1k-val.sh http://xxxxxx/challenges/LSVRC/2012/nnoupb/ILSVRC2012_img_val.tar

if [ ! -e ILSVRC2012_img_val.tar ]; then
    wget $1
fi
mkdir -p val
tar -xf ILSVRC2012_img_val.tar -C val
wget http://data.mxnet.io/models/imagenet/resnet/val.lst -O imagenet1k-val.lst

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MX_DIR=${CUR_DIR}/../../../

python ${CUR_DIR}/../../../tools/im2rec.py --resize 256 --quality 90 --num-thread 16 imagenet1k-val val/

rm -rf val
