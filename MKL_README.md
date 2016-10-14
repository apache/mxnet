# MKL2017 PLUGIN

MKL2017 is one INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.
This README shows user how to setup and install MKL2017 library with mxnet.


## Build/Install Instructions:
```
Download MKL:

```

## Build/Install MxNet
  1. Enable USE_MKL2017=1 in make/config.mk
  2. Run 'make -jX'
    2.1 Makefile will execute "prepare_mkl.sh" to download the mkl automatically
  3. Navigate into the python directory
  4. Run 'python setup.py install'
```
