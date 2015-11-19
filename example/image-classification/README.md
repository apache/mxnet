# Image Classification

This fold contains examples for image classifications. In this task, we assign
labels to an image with confidence scores, see the following figure for example ([source](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)):

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/image-classification.png
width=400/>

## How to use

First build mxnet by following the [guide](http://mxnet.readthedocs.org/en/latest/build.html)

### Train

Use `train_dataset.py` to train models on a particular dataset. For example:

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

- choose another intial learning rate, and decay it by 0.9 for every half epoch

  ```bash
  python train_mnist.py --lr .1 --lr-factor .9 --lr-factor-epoch .5
  ```

- train a convolution neural network on mnist by using GPU 0:

  ```bash
  python train_mnist.py --network lenet --gpus 0
  ```

  we can also use multiple GPUs by giving the list, e.g. `---gpus 0,1,3`

- uses `--help` to see more options

### Distributed Training

We can train a model using multiple machines.

- have a quick test on local machine by using two workers

  ```bash
  ../../tools/launch.py -n 2 python train_mnist.py --kv-store dist_sync
  ```

  here we can either use synchronized SGD `dist_sync` or use asynchronized SGD
  `dist_async`

- assume there are several ssh-able machines, and this mxnet folder is
  accessible on these machines (mounted as a NFS, see a tutorial for [Ubuntu](https://help.ubuntu.com/lts/serverguide/network-file-system.html)). To run a job on these machines, we
  first save their hostnames on a file, e.g.

  ```bash
  $ cat hosts
  172.30.0.172
  172.30.0.171
  ```

  then pass this file by `-H`

  ```bash
  ../../tools/launch.py -n 2 -H hosts python train_mnist.py --kv-store dist_sync
  ```

- If the mxnet folder is not available on other machines, we can first copy the mxnet
  library to this example folder


  ```bash
  cp -r ../../python/mxnet .
  cp -r ../../lib/libmxnet.so mxnet
  ```

  then synchronizing it to other machines' `/tmp/mxnet` before running

  ```bash
  ../../tools/launch.py -n 2 -H hosts --sync-dir /tmp/mxnet python train_mnist.py --kv-store dist_sync
  ```

See more launch options, e.g. by `Yarn`, and how to write a distributed training
program on this [tutorial](http://mxnet.readthedocs.org/en/latest/distributed_training.html)

### Predict

- [predict with pre-trained model](../notebooks/predict-with-pretrained-model.ipynb)
- more pre-trained models are provided on the [model gallery](https://github.com/dmlc/mxnet-model-gallerya).
- We can also easily to run the prediction on various devices, such as
[Android/iOS](http://dmlc.ml/mxnet/2015/11/10/deep-learning-in-a-single-file-for-smart-device.html)


### Use Your Own Datasets

There are two commonly used way to feed data into MXNet.

The first is packing all example into one or several compact `recordio`
files. See a step-by-step
[tutorial](https://mxnet.readthedocs.org/en/latest/python/io.html#create-dataset-using-recordio)
and the
[document](http://mxnet.readthedocs.org/en/latest/developer-guide/note_data_loading.html)
describing how it works.

*Note: A commonly mistake is forgetting shuffle the image list during packing. This will lead fail of training, eg. ```accuracy``` keeps 0.001 for several rounds.*

*Note: We will automatically download the small datasets such as `mnist` and `cifar10`*

The second way is for small datasets which can be easily loaded into memory. An
example is shown below:

```python
from sklearn.datasets import fetch_mldata
from sklearn.utils import shuffle
mnist = fetch_mldata('MNIST original', data_home="./mnist")
# shuffle data
X, y = shuffle(mnist.data, mnist.target)
# split dataset
train_data = X[:50000, :].astype('float32')
train_label = y[:50000]
val_data = X[50000: 60000, :].astype('float32')
val_label = y[50000:60000]
# Normalize data
train_data[:] /= 256.0
val_data[:] /= 256.0
# create a numpy iterator
batch_size = 100
train_iter = mx.io.NDArrayIter(train_data, train_label, batch_size=batch_size, shuffle=True)
val_iter = mx.io.NDArrayIter(val_data, val_label, batch_size=batch_size)
# create model as usual: model = mx.model.FeedForward(...)
model.fit(X=train_data, y=train_label)
```

## Results

In default we compile mxnet with CUDA7.5 + CUDNN 3

- `train_cifar10.py`

| 1 GTX 980 | 2 GTX 980 | 4 GTX 980 |
| --- | --- | --- |
| 842 img/sec | 1640 img/sec | 2943 img/sec |


- `train_imagenet.py` with `--network alexnet`

| 1 x GTX 980 | 2 x GTX 980  | 4 x GTX 980  |
| ----------- | ------------ | ------------ |
| 527 img/sec | 1030 img/sec | 1413 img/sec |

- `train_imagenet.py` with `--network inception-bn`

| 1 x GTX 980           | 2 x GTX 980            | 4 x GTX 980             |
| --------------------- | ---------------------- | ----------------------- |
| 97 img/sec (batch 32) | 178 img/sec (batch 64) | 357 img/sec (batch 128) |

For Inception-BN network, single model + single center test top-5 accuracy will
be round 90%.
