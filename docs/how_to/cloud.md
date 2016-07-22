# Cloud setup for MXNet

## Setup an AWS GPU Cluster from Scratch

In this document we give a step-by-step tutorial on how to set up Amazon AWS for
MXNet. In particular, we will address:

- [Use Amazon S3 to host data](#use-amazon-s3-to-host-data)
- [Setup EC2 GPU instance with all dependencies installed](#setup-an-ec2-gpu-instance)
- [Build and Run MXNet on a single machine](#build-and-run-mxnet-on-a-gpu-instance)
- [Setup an EC2 GPU cluster for distributed training](#setup-an-ec2-gpu-cluster)

### Use Amazon S3 to host data

Amazon S3 is distributed data storage, which is quite convenient for hosting large datasets. To use S3, we first get the
[AWS credentials](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html),
which includes an `ACCESS_KEY_ID` and a `SECRET_ACCESS_KEY`.

To use MXNet with S3, we must set the environment variables `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` properly. This can be done by adding the following two lines in
`~/.bashrc` (replacing the strings with the correct ones)

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

There are several ways to upload local data to S3. One simple way is using
[s3cmd](http://s3tools.org/s3cmd). For example:

```bash
wget http://data.dmlc.ml/mxnet/data/mnist.zip
unzip mnist.zip && s3cmd put t*-ubyte s3://dmlc/mnist/
```

### Set Up an EC2 GPU Instance

MXNet requires the following libraries

- C++ compiler with C++11 suports, such as `gcc >= 4.8`
- `CUDA` (`CUDNN` in optional) for GPU linear algebra
- `BLAS` (cblas, open-blas, atblas, mkl, or others) for CPU linear algebra
- `opencv` for image augmentations
- `curl` and `openssl` for read/write Amazon S3

Installing `CUDA` on EC2 instances requires some effort. Caffe has a nice
[tutorial](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN))
on how to install CUDA 7.0 on Ubuntu 14.04 (Note: we tried CUDA 7.5 on Nov 7
2015, but it is problematic.)

The rest can be installed by the package manager. For example, on Ubuntu:

```
sudo apt-get update
sudo apt-get install -y build-essential git libcurl4-openssl-dev libatlas-base-dev libopencv-dev python-numpy
```

We provide a public Amazon Machine Images, [ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178), with the above packages installed.


### Build and Run MXNet on a GPU Instance

The following commands build MXNet with CUDA/CUDNN, S3, and distributed
training.

```bash
git clone --recursive https://github.com/dmlc/mxnet
cd mxnet; cp make/config.mk .
echo "USE_CUDA=1" >>config.mk
echo "USE_CUDA_PATH=/usr/local/cuda" >>config.mk
echo "USE_CUDNN=1" >>config.mk
echo "USE_BLAS=atlas" >> config.mk
echo "USE_DIST_KVSTORE = 1" >>config.mk
echo "USE_S3=1" >>config.mk
make -j$(nproc)
```

To test whether everything is installed properly, we train a Convolutional neural network on MNIST using a GPU:

```bash
python tests/python/gpu/test_conv.py
```

If the MNISt data is placed on `s3://dmlc/mnist`, we can read the S3 data directly with the following command

```bash
sed -i.bak "s!data_dir = 'data'!data_dir = 's3://dmlc/mnist'!" tests/python/gpu/test_conv.py
```

Note: We can use `sudo ln /dev/null /dev/raw1394` to fix the opencv error `libdc1394 error: Failed to initialize libdc1394`.

### Set Up an EC2 GPU Cluster

A cluster consists of multiple machines. We can use the machine with MXNet
installed as the root machine for submitting jobs, and then launch several
slaves machine to run the jobs. For example, launch multiple instances using a
AMI, e.g.
[ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178),
with dependencies installed. There are two options:

1. Make all slaves' ports accessible (same for the root) by setting **type: All TCP**,
   **Source: Anywhere** in **Configure Security Group**

2. Use the same `pem` as the root machine to access all slave machines, and
   then copy the `pem` file into root machine's `~/.ssh/id_rsa`. If you do this, all slave machines are ssh-able from the root.

Now we run the previous CNN on multiple machines. Assume we are on a working
directory of the root machine, such as `~/train`, and MXNet is built as `~/mxnet`.

1. First pack the mxnet python library into this working directory for easy
  synchronization:

  ```bash
  cp -r ~/mxnet/python/mxnet .
  cp ~/mxnet/lib/libmxnet.so mxnet/
  ```

  And then copy the training program:

  ```bash
  cp ~/mxnet/example/image-classification/*.py .
  ```

2. Prepare a host file with all slaves's private IPs. For example, `cat hosts`

  ```bash
  172.30.0.172
  172.30.0.171
  ```

3. Assume there are 2 machines, then train the CNN using 2 workers:

  ```bash
  ../../tools/launch.py -n 2 -H hosts --sync-dir /tmp/mxnet python train_mnist.py --kv-store dist_sync
  ```

Note: Sometimes the jobs lingers at the slave machines even we pressed `Ctrl-c`
at the root node. We can kill them by

```bash
cat hosts | xargs -I{} ssh -o StrictHostKeyChecking=no {} 'uname -a; pgrep python | xargs kill -9'
```

Note: The above example is quite simple to train and therefore is not a good
benchmark for the distributed training. We may consider other [examples](https://github.com/dmlc/mxnet/tree/master/example/image-classification).

### More NOTE
#### Use multiple data shards
It is common to pack a dataset into multiple files, especially when working in a distributed environment. MXNet supports direct loading from multiple data shards. Simply put all the record files into a folder, and point the data path to the folder.

#### Use YARN, MPI, SGE
While ssh can be simple for cases when we do not have a cluster scheduling framework. MXNet is designed to be able to port to various platforms.  We also provide other scripts in [tracker](https://github.com/dmlc/dmlc-core/tree/master/tracker) to run on other cluster frameworks, including Hadoop(YARN) and SGE. Your contribution is more than welcomed to provide examples to run mxnet on your favorite distributed platform.
