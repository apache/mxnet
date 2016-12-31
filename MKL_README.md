# MKL2017 PLUGIN

MKL2017 is an INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.

MKL2017_ML is a subset of MKL2017 and only contains DNN acceleration feature

This README shows the user how to setup and install MKL2017 library with mxnet.

## Build/Install MXNet with MKL:

  1. Enable USE_MKL2017=1 in make/config.mk

    1.1 By default, MKL_2017_EXPRIEMENTAL=0. If setting MKL_2017_EXPRIEMENTAL=1, MKL buffer will be created and transferred between layers to achiever much higher performance.

    1.2 By default, USE_BLAS=atlas, MKLML_ROOT=/usr/local

      1.2.1 when excute make, Makefile will execute "prepare_mkl.sh" to download the MKL2017_ML library under <MKLML_ROOT>

      1.2.2 manually steps for download MKL2017_ML problem

        1.2.2.1 wget https://github.com/dmlc/web-data/raw/master/mxnet/mklml-release/mklml_lnx_<MKL VERSION>.tgz

        1.2.2.2 tar zxvf mklml_lnx_<MKL VERSION>.tgz
    
        1.2.2.3 cp -rf mklml_lnx_<MKL VERSION>/* <MKLML_ROOT>/

      1.2.2 If setting USE_BLAS=mkl, please navigate here to do a full MKL installation: https://registrationcenter.intel.com/en/forms/?productid=2558&licensetype=2     

  2. Run 'make -jX'
       
  3. Navigate into the python directory
  
  4. Set LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<MKLML_ROOT>/lib

  5. Run 'sudo python setup.py install'


