# Setup an AWS GPU Cluster from Stratch

In this documents we give a step-by-step tutorial on how to setup Amazon AWS for
MXNet. In particular, we will address:

- [Use Amazon S3 to host data]()
- [Setup EC2 GPU instance with all dependencies installed]()
- [Build and Run MXNet on a single machine with multiple GPU cards]()
- [Setup an EC2 GPU cluster for distributed training]()

## Use Amazon S3 to host data

Amazon S3 is distributed data storage, which is quite convenient for host large
scale datasets. In order to S3, we need first to get the
[AWS credentials](http://docs.aws.amazon.com/AWSSimpleQueueService/latest/SQSGettingStartedGuide/AWSCredentials.html)),
which includes a `ACCESS_KEY_ID` and a `SECRET_ACCESS_KEY`.

In order for MXNet to use S3, we only need to set the environment variables `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY` properly. For example, we can add the following two lines in
`~/.bashrc` (replace the strings with the correct ones)

```bash
export AWS_ACCESS_KEY_ID=AKIAIOSFODNN7EXAMPLE
export AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY
```

There are several ways to upload local data to S3. One simple way is using
[s3cmd](http://s3tools.org/s3cmd). For example:

```bash
wget http://webdocs.cs.ualberta.ca/~bx3/data/cifar10.zip
unzip cifar10.zip
s3cmd put cifar/* s3://dmlc/cifar10/
```

## Setup an EC2 GPU Instance

MXNet requires the following libraries

- C++ compiler with C++11 suports, such as `gcc >= 4.8`
- `CUDA` (`CUDNN` in optional) for GPU linear algebra
- `BLAS` (cblas, open-blas, atblas, mkl, or others) for CPU linear algebra
- `opencv` for image augmentations
- `curl` and `openssl` for read/write Amazon S3

Installing `CUDA` on EC2 instances needs a little bit effects. Caffe has a nice
[tutorial](https://github.com/BVLC/caffe/wiki/Install-Caffe-on-EC2-from-scratch-(Ubuntu,-CUDA-7,-cuDNN))
on how to install CUDA 7.0 on Ubuntu 14.04 (Note: we tried CUDA 7.5 on Nov 7
2015, but it is problematic.)

The reset can be installed by the package manager. For example, on Ubuntu:

```
sudo apt-get update
sudo apt-get install -y build-essential git libcurl4-openssl-dev libatlas-base-dev libopencv-dev
```

We provide a public Amazon Machine Images, [ami-12fd8178](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#LaunchInstanceWizard:ami=ami-12fd8178), with the above packages installed.


## Build and Run MXNet on a GPU Instance

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
make -j8
```

Test if every goes well:

```bash
TODO
```

Test S3
```bash
TODO
```

Note: if get error xxx, use

```bash
sudo ln /dev/null /dev/raw1394
```



## Setup an EC2 GPU Cluster


Configure Security Group

type: All TCP, Source Anywhere


cp ~/mxnet/example/distributed-training/*cifar* .


cat hosts | xargs -I{} ssh -o StrictHostKeyChecking=no {} 'uname -a; pgrep python | xargs kill -9'

single machine

INFO:root:Iter[0] Batch [10]	Speed: 101.93 samples/sec
INFO:root:Iter[0] Batch [20]	Speed: 83.47 samples/sec
INFO:root:Iter[0] Batch [30]	Speed: 83.53 samples/sec
INFO:root:Iter[0] Batch [40]	Speed: 83.63 samples/sec
INFO:root:Iter[0] Batch [50]	Speed: 83.86 samples/sec
INFO:root:Iter[0] Batch [60]	Speed: 83.39 samples/sec
INFO:root:Iter[0] Batch [70]	Speed: 83.50 samples/sec
INFO:root:Iter[0] Batch [80]	Speed: 83.43 samples/sec
INFO:root:Iter[0] Batch [90]	Speed: 83.46 samples/sec
