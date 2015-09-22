"""Create a Cifar data iterator.

This example shows how to create a iterator reading from recordio,
introducing image augmentations and using a backend thread to hide IO cost.
All you need to do is to set some parameters.
"""
import mxnet as mx

dataiter = mx.io.ImageRecordIter(
        # Dataset Paramter
        # Impulsary
        # indicating the data file, please check the data is already there
        path_imgrec="data/cifar/train.rec",
        # Dataset/Augment Paramter
        # Impulsary
        # indicating the image size after preprocessing
        data_shape=(3,28,28),
        # Batch Paramter
        # Impulsary
        # tells how many images in a batch
        batch_size=100,
        # Augmentation Parameter
        # Optional
        # when offers mean_img, each image will substract the mean value at each pixel
        mean_img="data/cifar/cifar10_mean.bin",
        # Augmentation Parameter
        # Optional
        # randomly crop a patch of the data_shape from the original image
        rand_crop=True,
        # Augmentation Parameter
        # Optional
        # randly mirror the image horizontally
        rand_mirror=True,
        # Augmentation Parameter
        # Optional
        # randomly shuffle the data
        shuffle=False,
        # Backend Parameter
        # Optional
        # Preprocessing thread number
        preprocess_threads=4,
        # Backend Parameter
        # Optional
        # Prefetch buffer size
        prefetch_buffer=4,
        # Backend Parameter,
        # Optional
        # Whether round batch,
        round_batch=True)

batchidx = 0
for data, label in dataiter:
    print("Batch", batchidx)
    print(label.asnumpy().flatten())
    batchidx += 1
