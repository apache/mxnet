# MXNet

[![Build Status](https://travis-ci.org/dmlc/mxnet.svg?branch=master)](https://travis-ci.org/dmlc/mxnet)
[![Documentation Status](https://readthedocs.org/projects/mxnet/badge/?version=latest)](https://readthedocs.org/projects/mxnet/?badge=latest)
[![GitHub Stats](https://img.shields.io/badge/github-stats-ff5500.svg)](http://githubstats.com/dmlc/mxnet)
[![Hex.pm](https://img.shields.io/hexpm/l/plug.svg)]()

This is a project that combines lessons and ideas we learnt from [cxxnet](https://github.com/dmlc/cxxnet), [minerva](https://github.com/dmlc/minerva) and [purine2](https://github.com/purine/purine2).
- The interface is designed in collaboration by authors of three projects.
- Nothing is yet working

# Guidelines
* Use google c style
* Put module header in [include](include)
* Depend on [dmlc-core](https://github.com/dmlc/dmlc-core)
* Doxygen comment every function, class and variable for the module headers
  - Ref headers in [dmlc-core/include](https://github.com/dmlc/dmlc-core/tree/master/include/dmlc)
  - Use the same style as dmlc-core
* Minimize dependency, if possible only depend on dmlc-core
* Macro Guard CXX11 code by
  - Try to make interface compile when c++11 was not avaialable(but with some functionalities pieces missing)
```c++
#include <dmlc/base.h>
#if DMLC_USE_CXX11
  // c++11 code here
#endif
```
  - Update the dependencies by
```
git submodule foreach --recursive git pull origin master
```
* For heterogenous hardware support (CPU/GPU). Hope the GPU-specific component could be isolated easily. That is too say if we use `USE_CUDA` macro to wrap gpu-related code, the macro should not be everywhere in the project.
