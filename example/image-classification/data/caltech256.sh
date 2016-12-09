#!/bin/bash

# This file download the caltech 256 dataset
# (http://www.vision.caltech.edu/Image_Datasets/Caltech256/), and split it into
# the train and val rec files.

# number of images per class for training
IMG_TRAIN=60

# download
if [ ! -e 256_ObjectCategories.tar ]; then
    wget http://www.vision.caltech.edu/Image_Datasets/Caltech256/256_ObjectCategories.tar
fi

# split into train and val set
tar -xf 256_ObjectCategories.tar
TRAIN_DIR=caltech_256_train
mkdir -p ${TRAIN_DIR}
for i in 256_ObjectCategories/*; do
    c=`basename $i`
    echo "spliting $c"
    mkdir -p ${TRAIN_DIR}/$c
    for j in `ls $i/*.jpg | shuf | head -n ${IMG_TRAIN}`; do
        mv $j ${TRAIN_DIR}/$c/
    done
done

# generate lst files
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MX_DIR=${CUR_DIR}/../../../
python ${MX_DIR}/tools/im2rec.py --list True --recursive True caltech256-train ${TRAIN_DIR}/
python ${MX_DIR}/tools/im2rec.py --list True --recursive True caltech256-val 256_ObjectCategories/
mv caltech256-train_train.lst caltech256-train.lst
rm caltech256-train_*
mv caltech256-val_train.lst caltech256-val.lst
rm caltech256-val_*

# generate rec files
python ${MX_DIR}/tools/im2rec.py --resize 256 --quality 95 --num-thread 16 caltech256-val 256_ObjectCategories/
python ${MX_DIR}/tools/im2rec.py --resize 256 --quality 95 --num-thread 16 caltech256-train ${TRAIN_DIR}/

# clean
rm -rf ${TRAIN_DIR} 256_ObjectCategories/
