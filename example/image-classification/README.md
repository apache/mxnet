# Image Classification

This fold contains examples for image classifications. In this task, we assign
labels to an image with confidence scores, see the following figure for example ([source](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)):

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/image-classification.png
width=400/>

## How to use

First build mxnet by following the [guide](http://mxnet.readthedocs.org/en/latest/build.html)

### Train

Use `train_dataset.py` to training models on a particular dataset. For example:


- train a MLP on mnist

  ```bash
  python train_mnist.py
  ```

- save the models for each epoch

  ```bash
  mkdir model; python train_mnist.py --model-prefix model/mnist
  ```

- restart training from the model saved at epoch 8

  ```bash
  python train_mnist.py --model-prefix model/mnist --load-epoch 8
  ```

- train a convolution neural network on mnist by using GPU 0:

  ```bash
  python train_mnist.py --network lenet --gpus 0
  ```

  we can also use multiple GPUs by `---gpus 0,1,3`

- uses `--help` to see more options

- Distributed training, e.g.using multiple GPU machines, is also support, refer to
  [Distributed Training](../distributed-training/) for how to launch the jobs.
  See more options by `--help`

### Predict

Pre-trained models are provided on the
[model gallery](https://github.com/dmlc/mxnet-model-gallerya).

We can also easily to run the prediction on various devices, such as
[Android/iOS](http://dmlc.ml/mxnet/2015/11/10/deep-learning-in-a-single-file-for-smart-device.html)


### Use Your Own Datasets

Please refer to the document
"[How to Create Dataset Using RecordIO](https://mxnet.readthedocs.org/en/latest/python/io.html#create-dataset-using-recordio)"
for a step-by-step tutorial.

Note: A commonly mistake is forgetting shuffle the image list during packing. This will lead fail of training, eg. ```accuracy``` keeps 0.001 for several rounds.

Note: We will automatically download the small datasets such as `mnist` and `cifar10`
