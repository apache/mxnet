# MKLDNN PLUGIN

MKLDNN is one INTEL released library to accelerate Deep Neural Network (DNN) applications on Intel architecture.
This README shows users how to setup mxnet with MKLDNN library.

## prepare MKLDNN Minimal Library
```
  cd <MXNET ROOTDIR>
  mkdir -p ./external/mkl
  wget https://github.com/intel/caffe/releases/download/self_containted_MKLGOLD/mklml_lnx_2017.0.0.20160801.tgz
  mv mklml_lnx_2017.0.0.20160801.tgz ./external/mkl
  cd external/mkl
  tar zxvf mklml_lnx_2017.0.0.20160801.tgz
  cd <MXNET ROOTDIR> 
```

## update config.mk
```
  USE_MKLDNN = 1 # set USE_MKLDNN on
  MKLDNN_ROOT = <MXNET ROOTDIR>/external/mkl/mklml_lnx_2017.0.0.20160801 # set MKLDNN ROOT PATH
```

## update LD_LIBRARY_PATH
```
  export LD_LIBRARY_PATH=<MXNET ROOTDIR>/external/mkl/mklml_lnx_2017.0.0.20160801/lib:$LD_LIBRARY_PATH
```

## build mxnet
```
  make -j8
```
