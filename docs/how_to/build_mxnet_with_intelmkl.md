# Building MXNet with MKL-based optimizations:

MXNet can be installed and used with several combinations of development 
tools and libraries on a variety of platforms. This tutorial provides one 
such recipe describing steps to build and install MXNet with Intel MKL on 
an Ubuntu-based system, version 16.04.

## Installing

1. Update your operating system and install dependencies:

   ```bash
    $ sudo apt-get update && apt-get install -y build-essential git libopencv-dev curl gcc libatlas-base-dev python python-pip python-dev python-opencv graphviz python-scipy python-sklearn
   ```

2. Delete any old or outdated versions of Intel MKL from `/tmp` before 
   downloading and copying the newest version to `/usr/local`:

   ```bash
   $ cd /tmp && rm -rf mklml_lnx_*   
   $ curl -L https://github.com/01org/mkl-dnn/releases/download/v0.10/mklml_lnx_2018.0.20170908.tgz | tar xz
   $ cp -a mklml_lnx_*/* /usr/local/.
   ```

3. Clone the latest version of MXNet from the Apache Incubator GitHub repository 
   and update submodules recursively:

   ```bash
   $ git clone --recursive git@github.com:apache/incubator-mxnet.git
   $ cd incubator-mxnet
   $ git submodule update --recursive
   ```

4. Build MXNet for CPU with support for the latest version of MKL:

   ```bash
   $ make -j $(nproc) USE_OPENCV=1 USE_MKL2017=1 USE_BLAS=atlas
   ```

5. Set your `PYTHONPATH` variable to location with the `incubator-mxnet/python` 
   submodule. For example, if you cloned it to the `/opt` directory, you'd update 
   or add `PYTHONPATH` as follows. Note that your regular `PATH` environment 
   variable should also be set to read where you cloned incubator-mxnet: 

   ```bash
   $ export PYTHONPATH=/opt/incubator-mxnet/python
   ```

6. Lastly, we can install some common Python tools for deep learning, and 
   verify that everything installed correctly by testing printing of some sample 
   output:     

   ```bash
   $ pip install --upgrade pip --user
   $ pip install --upgrade jupyter graphviz cython pandas bokeh matplotlib opencv-python requests --user
   $ python -c "import mxnet as mx;print((mx.nd.ones((2,3))*2).asnumpy());" 
   [[2.2.2.]
    [2.2.2.]]
    ```
  
## Benchmarks
  
A range of standard image classification benchmarks can be found under the 
  
```bash
example/image-classification
``` 
  
directory. We’ll focus on running a benchmark meant to test inference across 
a range of topologies. 
  
### Running Inference Benchmark
  
The provided `benchmark_score.py` will run a variety of standard topologies 
(AlexNet, Inception, ResNet, etc) at a range of batch sizes and report the 
image-per-second results. Prior to running, set the following environment 
variables for optimal performance:
  
#### For Intel® Xeon® Processors
  
```
export OMP_NUM_THREADS=$(($(grep 'core id' /proc/cpuinfo | sort -u | wc -l)*2))
```
   
   
#### For Intel® Xeon Phi™ Processors
   
```
export OMP_NUM_THREADS=$(($(grep 'core id' /proc/cpuinfo | sort -u | wc -l)))export KMP_AFFINITY=granularity=fine,compact,1,0
```
    
Then run the benchmark with:
    
```
$ python example/image-classification/benchmark_score.py
 ```
     
If everything installed correctly, we should see the image-per-second numbered 
output for the different topologies and batch sizes:
     
```console
INFO:root:network: alexnet
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: XXX
INFO:root:batch size  2, image/sec: XXX
...
INFO:root:batch size 32, image/sec: XXX
INFO:root:network: vgg
INFO:root:device: cpu(0)
INFO:root:batch size  1, image/sec: XXX
```
     
 
[Compile with MKL-DNN performance library ]:compile_ml_libs 