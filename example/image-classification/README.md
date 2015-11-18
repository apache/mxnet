# Image Classification

This fold contains examples for image classifications. In this task, we assign
labels to an image with a confidence scores. For example ([source](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)):

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/image-classification.png
width=400/>

## How to use

- First build mxnet by following the [guide](http://mxnet.readthedocs.org/en/latest/build.html)

- Use `train_dataset.py` to training models on various dataset. For example,
  train a MLP on mnist

  ```bash
  python train_mnist.py
  ```

  or train a convetnet on mnist using GPU 0

  ```bash
  python train_mnist.py --network lenet --gpus 0
  ```

  See more options

  ```bash
  python train_mnist.py --help
  ```

- Pre-trained models are also provided.

## More

- [Distributed training](../distributed-training/)
- [Run prediction on mobiles](http://dmlc.ml/mxnet/2015/11/10/deep-learning-in-a-single-file-for-smart-device.html)
