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
