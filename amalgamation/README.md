MXNet Amalgamation
==================
This folder contains a amalgamation generation script to generate the entire mxnet library into one file.
Currently it supports generation for [predict API](../include/mxnet/c_predict_api.h),
which allows you to run prediction in platform independent way.

How to Generate the Amalgamation
--------------------------------
Type ```make``` will generate the following files
- mxnet_predict-all.cc
  - The file you can used to compile predict API
- ../lib/libmxnet_predict.so
  - The dynamic library generated for prediction.

You can also checkout the [Makefile](Makefile)

Dependency
----------
The only dependency is a BLAS library.

Acknowledgement
---------------
This module is created by [Jack Deng](https://github.com/jdeng).
