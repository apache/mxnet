mshadow: Matrix Shadow
======
[![Build Status](https://travis-ci.org/dmlc/mshadow.svg?branch=master)](https://travis-ci.org/dmlc/mshadow)

MShadow is a lightweight CPU/GPU Matrix/Tensor Template Library in C++/CUDA. The goal of mshadow is to support ***efficient***,
***device invariant*** and ***simple*** tensor library for machine learning project that aims for maximum performance and control, while also emphasize simplicity.

MShadow also provides interface that allows writing Multi-GPU and distributed deep learning programs in an easy and unified way.

* [Contributors](https://github.com/tqchen/mshadow/graphs/contributors)
* [Tutorial](guide)
* [Documentation](doc)
* [Parameter Server Interface for GPU Tensor](guide/mshadow-ps)

Features
--------
* Efficient: all the expression you write will be lazily evaluated and compiled into optimized code
  - No temporal memory allocation will happen for expression you write
  - mshadow will generate specific kernel for every expression you write in compile time.
* Device invariant: you can write one code and it will run on both CPU and GPU
* Simple: mshadow allows you to write machine learning code using expressions.
* Whitebox: put a float* into the Tensor struct and take the benefit of the package, no memory allocation is happened unless explicitly called
* Lightweight library: light amount of code to support frequently used functions in machine learning
* Extendable: user can write simple functions that plugs into mshadow and run on GPU/CPU, no experience in CUDA is required.
* MultiGPU and Distributed ML: mshadow-ps interface allows user to write efficient MultiGPU and distributed programs in an unified way.

Version
-------
* This version mshadow-2.x, there are a lot of changes in the interface and it is not backward compatible with mshadow-1.0
  - If you use older version of cxxnet, you will need to use the legacy mshadow code
* For legacy code, refer to [Here](https://github.com/tqchen/mshadow/releases/tag/v1.1)
* Change log in [CHANGES.md](CHANGES.md)

Projects Using MShadow
----------------------
* [MXNet: Efficient and Flexible Distributed Deep Learning Framework](https://github.com/dmlc/mxnet)
* [CXXNet: A lightweight  C++ based deep learnig framework](https://github.com/dmlc/cxxnet)
