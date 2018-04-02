### NNPACK for Multi-Core CPU Support in MXNet
[NNPACK](https://github.com/Maratyszcza/NNPACK) is an acceleration package
for neural network computations, which can run on x86-64, ARMv7, or ARM64 architecture CPUs.
Using NNPACK, higher-level libraries like _MXNet_ can speed up
the execution on multi-core CPU computers, including laptops and mobile devices.

_MXNet_ supports NNPACK for forward propagation (inference only) in convolution, max-pooling, and fully-connected layers.
In this document, we give a high level overview of how to use NNPACK with _MXNet_.


### Conditions
The underlying implementation of NNPACK utilizes several acceleration methods,
including [fft](https://arxiv.org/abs/1312.5851) and [winograd](https://arxiv.org/abs/1509.09308).
These algorithms work better on some special `batch size`, `kernel size`, and `stride` settings than on other,
so depending on the context, not all convolution, max-pooling, or fully-connected layers can be powered by NNPACK.
When favorable conditions for running NNPACKS are not met,
_MXNet_ will fall back to the default implementation automatically.  

NNPACK only supports Linux and OS X systems. Windows is not supported at present.
The following table explains under which conditions NNPACK will work.

| operation      | conditions |
|:---------      |:---------- |
|convolution     |2d convolution `and` no-bias=False `and` dilate=(1,1) `and` num_group=1 `and` batch-size = 1 or batch-size > 1 && stride = (1,1);|
|pooling         | max-pooling `and` kernel=(2,2) `and` stride=(2,2) `and` pooling_convention=full    |
|fully-connected| without any restrictions |

### Build/Install NNPACK with MXNet

If the trained model meets some conditions of using NNPACK,
you can build MXNet with NNPACK support.
Follow these simple steps:  
* Build NNPACK shared library with the following commands. _MXNet_ will link NNPACK dynamically.

Note: The following NNPACK installation instructions have been tested on Ubuntu 14.04 and 16.04.

```bash

# Install Pip
$ sudo apt-get update
$ sudo apt-get install -y python-pip
$ sudo pip install --upgrade pip

# Install Peach
$ git clone https://github.com/Maratyszcza/PeachPy.git
$ cd PeachPy
$ sudo pip install --upgrade -r requirements.txt
$ python setup.py generate
$ sudo pip install --upgrade .

# Install Ninja Build System
$ sudo apt-get install ninja-build
$ pip install ninja-syntax

# Build NNPack shared library
$ cd ~
$ git clone --recursive https://github.com/Maratyszcza/NNPACK.git
$ cd NNPACK
# Latest NNPACK do not support building NNPACK as shared library using --enable-shared flag
# Reset to commit that supports it.
$ git reset --hard 9c6747d7b80051b40e6f92d6828e2ed997529cd2
$ git submodule init && git submodule update --recursive
$ python ./configure.py --enable-shared
$ ninja
$ cd ~

```

* Set lib path of NNPACK as the environment variable, e.g. `export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$YOUR_NNPACK_INSTALL_PATH/lib`
* Add the include file of NNPACK and its third-party to  `ADD_CFLAGS` in config.mk, e.g. `ADD_CFLAGS = -I$(YOUR_NNPACK_INSTALL_PATH)/include/ -I$(YOUR_NNPACK_INSTALL_PATH)/third-party/pthreadpool/include/`
* Set `USE_NNPACK = 1` in config.mk.
* Build MXNet from source following the [install guide](http://mxnet.io/install/index.html).

### NNPACK Performance

Though not all convolutional, pooling, and fully-connected layers can make full use of NNPACK,
for some popular models it provides significant speedups. These include the most popular image recognition networks: Alexnet, VGG, and Inception-bn.

To benchmark NNPACK, we use `example/image-classification/benchmark_score.py`(changed with  more range of batch-size). We use CPU e5-2670, MXNET_CPU_NNPACK_NTHREADS=4.

build MXNet without NNPACK, the log is:
```
INFO:root:network: alexnet
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 6.389429
INFO:root:batch size  2, image/sec: 7.961457
INFO:root:batch size  4, image/sec: 8.950112
INFO:root:batch size  8, image/sec: 9.578176
INFO:root:batch size 16, image/sec: 9.701248
INFO:root:batch size 32, image/sec: 9.839940
INFO:root:batch size 64, image/sec: 10.075369
INFO:root:batch size 128, image/sec: 10.053556
INFO:root:batch size 256, image/sec: 9.972228
INFO:root:network: vgg
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 1.223822
INFO:root:batch size  2, image/sec: 1.322814
INFO:root:batch size  4, image/sec: 1.383586
INFO:root:batch size  8, image/sec: 1.402376
INFO:root:batch size 16, image/sec: 1.415972
INFO:root:batch size 32, image/sec: 1.428377
INFO:root:batch size 64, image/sec: 1.443987
INFO:root:batch size 128, image/sec: 1.427531
INFO:root:batch size 256, image/sec: 1.435279
```

build MXNet with NNPACK, log is:

```
INFO:root:network: alexnet
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 19.027215
INFO:root:batch size  2, image/sec: 12.879975
INFO:root:batch size  4, image/sec: 17.424076
INFO:root:batch size  8, image/sec: 21.283966
INFO:root:batch size 16, image/sec: 24.469325
INFO:root:batch size 32, image/sec: 25.910348
INFO:root:batch size 64, image/sec: 27.441672
INFO:root:batch size 128, image/sec: 28.009156
INFO:root:batch size 256, image/sec: 28.918950
INFO:root:network: vgg
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: 3.980907
INFO:root:batch size  2, image/sec: 2.392069
INFO:root:batch size  4, image/sec: 3.610553
INFO:root:batch size  8, image/sec: 4.994450
INFO:root:batch size 16, image/sec: 6.396612
INFO:root:batch size 32, image/sec: 7.614288
INFO:root:batch size 64, image/sec: 8.826084
INFO:root:batch size 128, image/sec: 9.193653
INFO:root:batch size 256, image/sec: 9.991472
```

The results show that NNPACK can confer a speedup of about 2X~7X as compared to the original _MXNet_ CPU implementation.

### Tips

NNPACK aims to provide high-performance implementations of some layers for multi-core CPUs, so you can easily set the thread number by changing the environmental variable `MXNET_CPU_NNPACK_NTHREADS`. However, we found that the performance is not proportional to the number of threads, and suggest using 4~8 threads when using NNPACK.
