# Intel(R) Math Kernel Library Optimizations for Machine Learning

MKL2017 is an INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.

MKL2017_ML is a subset of MKL2017 and only contains DNN acceleration features, MKL2017 release cycle is longer than MKL2017_ML and MKL2017_ML support latest features.

[Intel(R) Math Kernel Library for Deep Neural Networks (Intel(R) MKL-DNN)](https://github.com/01org/mkl-dnn) is a new open source performance library specially designed for accelerating Deep Learning (DL) applications on Intel(R) architecture. 

Intel MKL-DNN includes functionality similar to Intel(R) Math Kernel Library (Intel(R) MKL) 2017, and adds several new optimizations for Deep Learning workloads.

This README shows the user how to setup and install MXNet with MKL2017 and the newer MKL-DNN. Please choose one, we cannot build with both the options. 

* Quick Start for MXNet with MKL2017
  ``` bash
  # this will install MKLDNN under /usr/local by default. Please refer to detailed instructions for custom options.
  sudo make -j$(nproc) USE_MKL2017=1 USE_MKL2017_EXPERIMENTAL=1
  cd python
  sudo python setup.py install
  ```
* Quick Start for MXNet with MKL-DNN
  ``` bash
  # MKL-DNN uses cmake, please ensure you have installed it.
  # this will install MKLDNN under /usr/local by default. Please refer to detailed instructions for custom options.
  sudo make -j$(nproc) USE_MKLDNN=1
  cd python
  sudo python setup.py install
  ```


## Build/Install MXNet with MKL2017:
-------------

  1. Enable USE_MKL2017=1 in make/config.mk

    1.1 By default, MKL_2017_EXPRIEMENTAL=0. If setting MKL_2017_EXPRIEMENTAL=1, MKL buffer will be created and transferred between layers to achiever much higher performance.

    1.2 By default, MKLML_ROOT=/usr/local, MKL2017_ML will be used

      1.2.1 when excute make, Makefile will execute "prepare_mkl.sh" to download the MKL2017_ML library under <MKLML_ROOT>

      1.2.2 manually steps for download MKL2017_ML problem

        1.2.2.1 wget https://github.com/dmlc/web-data/raw/master/mxnet/mklml-release/mklml_lnx_<MKL VERSION>.tgz

        1.2.2.2 tar zxvf mklml_lnx_<MKL VERSION>.tgz
    
        1.2.2.3 cp -rf mklml_lnx_<MKL VERSION>/* <MKLML_ROOT>/

      1.2.3 Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLML_ROOT/lib

    1.3 If setting USE_BLAS=mkl

      1.3.1 mshadow can also utilize mkl blas function in mklml package  

    1.4 MKL version compatibility
        
        1.3.2.1 If you already have MKL installed and MKLROOT being set in your system, by default, it will not attempt to download the latest mklml package unless you unset MKLROOT. 

  2. Run 'make -jX'
       
  3. Navigate into the python directory
  
  4. Run 'sudo python setup.py install'

## Build/Install MXNet with MKL-DNN:
-------------

  1. Enable USE_MKLDNN=1 in make/config.mk

     - MKLDNN_ROOT option in make/config.mk allows user to choose install folder for MKLDNN. By default it is set to /usr/local, hence requires sudo. If set to empty, MKLDNN will be installed under external/mkldnn/install folder. 

     - when you excute make, Makefile will execute "prepare_mkldnn.sh" to download and build MKLDNN with mklml under external/mkldnn folder.

     - If you choose to install MKLDNN in custom folder, please set 

         export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$MKLDNN_ROOT/lib

     - If you set USE_BLAS=mkl, mshadow can utilize mkl blas function in MKLDNN mklml package  

  2. Run 'make -jX'
       
  3. Navigate into the python directory
  
  4. Run 'sudo python setup.py install'


