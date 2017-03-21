# Image Classification

In this tutorial, we assign
labels to an image with confidence scores. The following figure ([source](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)) shows an example:

<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/image-classification.png
width=600/>

Get the source code for the tutorial from [GitHub](https://github.com/dmlc/mxnet/tree/master/example/image-classification).


## Training

To train models on a particular dataset, use `train_dataset.py`. For example:

- To train an MLP on mnist, use this command:

```bash
  python train_mnist.py
```

- To save the models for each epoch, use this command:

```bash
  mkdir model; python train_mnist.py --model-prefix model/mnist
```

- To restart training from the model saved at epoch 8, use this command:

```bash
  python train_mnist.py --model-prefix model/mnist --load-epoch 8
```

- To choose another initial learning rate, and decay it by 0.9 for every half epoch, use this command:

```bash
  python train_mnist.py --lr .1 --lr-factor .9 --lr-factor-epoch .5
```

- To train a convolutional neural network on mnist by using GPU 0, use this command:

```bash
  python train_mnist.py --network lenet --gpus 0
```

- To use multiple GPUs, specify the list; for example: `---gpus 0,1,3.`

- To see more options, use `--help`.

## Distributed Training

To speed training, train a model using multiple computers.

* Quickly test distributed training on your local computer by using two workers:

```bash
  ../../tools/launch.py -n 2 python train_mnist.py --kv-store dist_sync
```

You can use either synchronous SGD `dist_sync` or asynchronous SGD
  `dist_async`.

* If you have several computers that you can connect to using SSH, and if this mxnet folder is
  accessible on these computers (is mounted as an NFS; see a tutorial for [Ubuntu](https://help.ubuntu.com/lts/serverguide/network-file-system.html)), run a job on these computers, first by saving their hostnames on a file, for example:

```bash
  $ cat hosts
  172.30.0.172
  172.30.0.171
```

* Then pass this file using `-H`:

```bash
  ../../tools/launch.py -n 2 -H hosts python train_mnist.py --kv-store dist_sync
```

* If the mxnet folder isn't available on the other computers, copy the mxnet
  library to this example folder:


```bash
  cp -r ../../python/mxnet .
  cp -r ../../lib/libmxnet.so mxnet
```

Then synchronize the folder to other the other computers `/tmp/mxnet` before running:

```bash
  ../../tools/launch.py -n 2 -H hosts --sync-dir /tmp/mxnet python train_mnist.py --kv-store dist_sync
```

For more launch options, for example, using `YARN`, and information about how to write a distributed training
program, see this [tutorial](http://mxnet.io/how_to/multi_devices.html).

## Generating Predictions
You have several options for generating predictions:

- Use a [pre-trained model](http://mxnet.io/tutorials/python/predict_imagenet.html). More pre-trained models are provided in the [model gallery](https://github.com/dmlc/mxnet-model-gallery).
- Use your own datasets.
- You can also easily run the prediction on various devices, such as
[Android/iOS](http://dmlc.ml/mxnet/2015/11/10/deep-learning-in-a-single-file-for-smart-device.html).


### Using Your Own Datasets

There are two ways to feed data into MXNet:

- Pack all examples into one or more compact `recordio` files. For more information, see this [step-by-step tutorial](http://mxnet.io/api/python/io.html#create-a-dataset-using-recordio) and [documentation](http://mxnet.io/architecture/note_data_loading.html). Avoid the common mistake of neglecting to shuffle the image list during packing. This causes training to fail. For example, ```accuracy``` keeps 0.001 for several rounds.

	**Note:** We automatically download the small datasets, such as `mnist` and `cifar10`.

- For small datasets, which can be easily loaded into memory, here is an example:

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
		model.fit(X = train_iter, eval_data = val_iter)
```

## Improving Performance

The following factors can significantly improve performance:

* A fast back end. A fast BLAS library, e.g., openblas, atlas,
and mkl, is necessary only if you are using a CPU processor. For Nvidia GPUs, we strongly
recommend using CUDNN.
* Input data:

	- Data format. Use the `rec` format.


	- A number of threads used for decoding. By default, MXNet uses four CPU threads for decoding images, which
    can often decode more than 1 Kb images per second. If you are using a low-end CPU or
    very powerful GPUs, you
    can increase the number of threads .


	- Data storage location. Any local or distributed file system (HDFS, Amazon
    S3) should be fine. If multiple computers read the
    data from the network shared file system (NFS) at the same time, however, you might encounter a problem.


	- Batch size. We recommend using the largest size that the GPU memory can accommodate. A value that is too large might slow down convergence. A safe batch size for CIFAR 10 is approximately 200; for ImageNet
  1K, the batch size can exceed 1 Kb.


* If you are using more than one GPU, the right `kvstore`. For more information, see
  [this guide](http://mxnet.io/how_to/multi_devices.html#distributed-training-with-multiple-machines).


	- For a single computer, the default `local` is often sufficient. For models bigger than 100 MB, such as AlexNet
  and VGG, you might want
  to use `local_allreduce_device`.  `local_allreduce_device` uses more GPU memory than
  other options.



	- For multiple computers, we recommend trying to use `dist_sync` first. If the
  model is very large or if you use a large number of computers, you might want to use `dist_async`.

## Results

- Computers

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

  <img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/inception-with-bn-cifar10.png width=400px/>

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

  - `10 x g2.8x` : `hosts` contains the private IPs of the 10 computers

```bash
  ../../tools/launch.py -H hosts -n 10 --sync-dir /tmp/mxnet  \
      python train_imagenet.py --batch-size 144 --lr 0.05 --lr-factor .94 \
        --gpus 0,1,2,3 --num-epoch 60 --network inception-bn \
        --kv-store dist_sync \
        --data-dir s3://dmlc/ilsvrc12/  --model-prefix s3://dmlc/model/ilsvrc12
```

  **Note:** Occasional instability in Amazon S3 might cause training to hang or generate frequent errors, preventing downloading data to `/mnt` first.

- Accuracy vs. epoch ([the interactive figure](https://docs.google.com/spreadsheets/d/1AEesHjWUZOzCN0Gp_PYI1Cw4U1kZMKot360p9Fowmjw/pubchart?oid=1740787404&format=interactive)):

  	<img src=https://raw.githubusercontent.com/dmlc/web-data/master/mxnet/image/inception-with-bn-imagnet1k.png width=400px/>

## Next Steps
* [MXNet tutorials index](http://mxnet.io/tutorials/index.html)
