<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Run-Time Feature detection / Library info

```eval_rst
.. currentmodule:: mxnet.runtime
```

## Overview

The libinfo functionality allows to check for compile-time features supported by the library.

### Example usage

```
In []: import mxnet as mx
   ...: import mxnet.runtime
   ...: fs = mx.runtime.Features()

In []: fs
Out[]: [✖ CUDA, ✖ CUDNN, ✖ NCCL, ✖ CUDA_RTC, ✖ TENSORRT, ✔ CPU_SSE, ✔ CPU_SSE2, ✔ CPU_SSE3, ✔ CPU_SSE4_1, ✔ CPU_SSE4_2, ✖ CPU_SSE4A, ✔ CPU_AVX, ✖ CPU_AVX2, ✖ OPENMP, ✖ SSE, ✔ F16C, ✖ JEMALLOC, ✔ BLAS_OPEN, ✖ BLAS_ATLAS, ✖ BLAS_MKL, ✖ BLAS_APPLE, ✔ LAPACK, ✖ MKLDNN, ✔ OPENCV, ✖ CAFFE, ✖ PROFILER, ✖ DIST_KVSTORE, ✖ CXX14, ✔ SIGNAL_HANDLER, ✔ DEBUG]

In []: fs.keys()
Out[]: odict_keys(['CUDA', 'CUDNN', 'NCCL', 'CUDA_RTC', 'TENSORRT', 'CPU_SSE', 'CPU_SSE2', 'CPU_SSE3', 'CPU_SSE4_1', 'CPU_SSE4_2', 'CPU_SSE4A', 'CPU_AVX', 'CPU_AVX2', 'OPENMP', 'SSE', 'F16C', 'JEMALLOC', 'BLAS_OPEN', 'BLAS_ATLAS', 'BLAS_MKL', 'BLAS_APPLE', 'LAPACK', 'MKLDNN', 'OPENCV', 'CAFFE', 'PROFILER', 'DIST_KVSTORE', 'CXX14', 'SIGNAL_HANDLER', 'DEBUG'])

In []: type(fs['CUDA'])
Out[]: mxnet.runtime.Feature

In []: fs['CUDA'].enabled
Out[]: False

In []: fs.is_enabled('CPU_SSE')
Out[]: True

In []: fs.is_enabled('CUDA')
Out[]: False

In []: features = mx.runtime.feature_list()

In []: features
Out[]:
[✖ CUDA,
 ✖ CUDNN,
 ✖ NCCL,
 ✖ CUDA_RTC,
 ✖ TENSORRT,
 ✔ CPU_SSE,
 ✔ CPU_SSE2,
 ✔ CPU_SSE3,
 ✔ CPU_SSE4_1,
 ✔ CPU_SSE4_2,
 ✖ CPU_SSE4A,
 ✔ CPU_AVX,
 ✖ CPU_AVX2,
 ✖ OPENMP,
 ✖ SSE,
 ✔ F16C,
 ✖ JEMALLOC,
 ✔ BLAS_OPEN,
 ✖ BLAS_ATLAS,
 ✖ BLAS_MKL,
 ✖ BLAS_APPLE,
 ✔ LAPACK,
 ✖ MKLDNN,
 ✔ OPENCV,
 ✖ CAFFE,
 ✖ PROFILER,
 ✖ DIST_KVSTORE,
 ✖ CXX14,
 ✔ SIGNAL_HANDLER,
 ✔ DEBUG]

In []: type(features)
Out[]: list

In []: type(features[0])
Out[]: mxnet.runtime.Feature
```

```eval_rst
.. autosummary::
    :nosignatures:

```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.runtime
    :members:
```

<script>auto_index("api-reference");</script>
