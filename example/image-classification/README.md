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
- more pre-trained models are provided on the [model gallery](https://github.com/dmlc/mxnet-model-gallery).
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

## Note on Performance

The following factors may significant affect the performance:

1. Use a fast backend. A fast BLAS library, e.g. openblas, altas,
and mkl, is necessary if only using CPU. While for Nvidia GPUs, we strongly
recommend to use CUDNN.
2. Three important things for the input data:
  1. data format. If you are using the `rec` format, then everything should be
    fine.
  2. decoding. In default MXNet uses 4 CPU threads for decoding the images, which
    are often able to decode over 1k images per second. You
    may increase the number of threads if either you are using a low-end CPU or
    you GPUs are very powerful.
  3. place to store the data. Any local or distributed filesystem (HDFS, Amazon
    S3) should be fine. There may be a problem if multiple machines read the
    data from the network shared filesystem (NFS) at the same time.
3. Use a large batch size. We often choose the largest one which can fit into
  the GPU memory. But a too large value may slow down the convergence. For
  example, the safe batch size for CIFAR 10 is around 200, while for ImageNet
  1K, the batch size can go beyond 1K.
4. Choose the proper `kvstore` if using more than one GPU. (See
  [doc/developer-guide/multi_node.md](../../doc/developer-guide/multi_node.md)
  for more information)
  1. For a single machine, often the default `local` is good enough. But you may want
  to use `local_allreduce_device` for models with size >> 100MB such as AlexNet
  and VGG. But also note that `local_allreduce_device` takes more GPU memory than
  others.
  2. For multiple machines, we recommend to try `dist_sync` first. But if the
  model size is quite large or you use a large number of machines, you may want to use `dist_async`.

## Results

- Machines

  | name | hardware | software |
  | --- | --- | --- |
  | GTX980 | Xeon E5-1650 v3, 4 x GTX 980 | GCC 4.8, CUDA 7.5, CUDNN 3 |
  | TitanX | dual Xeon E5-2630 v3, 4 x GTX Titan X | GCC 4.8, CUDA 7.5, CUDNN 3 |
  | EC2-g2.8x | Xeon E5-2670, 2 x GRID K520, 10G Ethernet | GCC 4.8, CUDA 7.5, CUDNN 3 |

- Datasets

  | name | class | image size | training | testing |
  | ---- | ----: | ---------: | -------: | ------: |
  | CIFAR 10 | 10 | 28 × 28 × 3 | 60,000  | 10,000 |
  | ILSVRC 12 | 1,000 | 227 × 227 × 3 | 1,281,167 | 50,000 |

### CIFAR 10

- Command

```bash
python train_cifar10.py --batch-size 128 --lr 0.1 --lr-factor .94 --num-epoch 50
```

- Performance:

  | 1 GTX 980 | 2 GTX 980 | 4 GTX 980 |
  | --- | --- | --- |
  | 842 img/sec | 1640 img/sec | 2943 img/sec |

- Accuracy vs epoch ([interactive figure](https://docs.google.com/spreadsheets/d/1kV2aDUXNyPn3t5nj8UdPA61AdRF4_w1UNmxaqu-cRBA/pubchart?oid=761035336&format=interactive)):

  <img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/inception-with-bn-cifar10.png width=600px/>

### ILSVRC 12

<!-- #### Alexnet -->

<!-- `train_imagenet.py` with `--network alexnet` -->

<!-- - time for one epoch: -->

<!--   | 1 x GTX 980 | 2 x GTX 980  | 4 x GTX 980  | -->
<!--   | ----------- | ------------ | ------------ | -->
<!--   | 2,413 sec | 1,244 sec | 906 sec | -->

#### VGG

`train_imagenet.py` with `--network vgg`

- Performance

  | Cluster | # machines | # GPUs | batch size | kvstore | epoch time |
  | --- | --- | --- | --- | --- | ---: |
  | TitanX | 1 | 1 | 96 | `none` | 14,545 |
  | - | - | 2 | - | `local` | 19,692 |
  | - | - | 4 | - | - | 20,014 |
  | - | - | 2 | - | `local_allreduce_device` | 9,142 |
  | - | - | 4 | - | - | 8,533 |
  | - | - | - | 384 | - | 5,161 |

#### Inception with Batch Normalization

`train_imagenet.py` with `--network inception-bn`

- Performance

  | Cluster | # machines | # GPUs | batch size | kvstore | epoch time |
  | --- | --- | --- | --- | --- | ---: |
  | GTX980 | 1 | 1 |  32 | `local` | 13,210 |
  | - | - | 2 |  64 | - | 7,198 |
  | - | - | 3 |  128 | - | 4,952 |
  | - | - | 4 |  - | - | 3,589 |
  | TitanX | 1 | 1 | 128 | `none` | 10,666 |
  | - | - | 2 | - | `local` | 5,161 |
  | - | - | 3 | - | - | 3,460 |
  | - | - | 4 | - | - | 2,844 |
  | - | - | - | 512 | - | 2,495 |
  | EC2-g2.8x | 1 | 4 | 144 |  `local` | 14,203 |
  | - | 10 | 40 | 144 |  `dist_sync` | 1,422 |

- Convergence

  - `single machine` :

  ```bash
  python train_imagenet.py --batch-size 144 --lr 0.05 --lr-factor .94 \
      --gpus 0,1,2,3 --num-epoch 60 --network inception-bn \
      --data-dir ilsvrc12/ --model-prefix model/ilsvrc12
  ```

  - `10 x g2.8x` : `hosts` contains the private IPs of the 10 machines

  ```bash
  ../../tools/launch.py -H hosts -n 10 --sync-dir /tmp/mxnet  \
      python train_imagenet.py --batch-size 144 --lr 0.05 --lr-factor .94 \
        --gpus 0,1,2,3 --num-epoch 60 --network inception-bn \
        --kv-store dist_sync \
        --data-dir s3://dmlc/ilsvrc12/  --model-prefix s3://dmlc/model/ilsvrc12
  ```

  *Note: S3 is unstable sometimes, if your training hangs or getting error
   freqently, you cant download data to `/mnt` first*

  Accuracy vs epoch ([the interactive figure](https://docs.google.com/spreadsheets/d/1AEesHjWUZOzCN0Gp_PYI1Cw4U1kZMKot360p9Fowmjw/pubchart?oid=1740787404&format=interactive)):

  <img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/inception-with-bn-imagnet1k.png width=600px/>
