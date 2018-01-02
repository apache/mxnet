# Full MKL Installation

## Build/Install MXNet with a full MKL installation:
Installing and enabling the full MKL installation enables MKL support for all operators under the linalg namespace.

  1. Download and install the latest full MKL version following instructions on the [intel website.](https://software.intel.com/en-us/articles/intel-mkl-111-install-guide)

  2. Set USE_BLAS=mkl in make/config.mk

        1.1 Set ADD_LDFLAGS=-L<path/to/mkl/lib/folder> (ex. ADD_LDFLAGS=-L/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/lib)

        1.1 Set ADD_CFLAGS=-I<path/to/mkl/include/folder> (ex. ADD_CFLAGS=-L/opt/intel/compilers_and_libraries_2018.0.128/linux/mkl/include)

  3. Run 'make -j ${nproc}'

  4. Navigate into the python directory

  5. Run 'sudo python setup.py install'

# MKL2017 PLUGIN

MKL2017 is an INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.

MKL2017_ML is a subset of MKL2017 and only contains DNN acceleration feature, MKL2017 release cycle is longer then MKL2017_ML and MKL2017_ML support latest feature

This README shows the user how to setup and install MKL2017 library with mxnet.

## Build/Install MXNet with MKL:

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


