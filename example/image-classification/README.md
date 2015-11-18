# Image Classification

This fold contains examples for image classifications. In this task, we assign
labels to an image with a confidence scores. For example ([source](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)):

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/image-classification.png
width=400/>

## How to use

- First build mxnet by following the [guide](http://mxnet.readthedocs.org/en/latest/build.html)
- [network](network/) contains various neural networks
- Other directories, e.g.~[mnist](mnist/), contain programs to train models on a
  particular dataset. e.g.

  ```bash
  cd mnist; python train_lenet.py
  ```

- pre-trained models are also provided.

## More

- [Distributed training](../distributed-training/)
- [Run prediction on mobiles](http://dmlc.ml/mxnet/2015/11/10/deep-learning-in-a-single-file-for-smart-device.html)
