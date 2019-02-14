# Run-Time Feature detection / Library info

```eval_rst
.. currentmodule:: mxnet.runtime
```

## Overview

The libinfo functionality allows to check for compile-time features supported by the library.

### Example usage

```
In [1]: import mxnet as mx
   ...: import mxnet.runtime
   ...: fs = mx.runtime.Features()

In [2]: fs
Out[2]: [✖ CUDA, ✖ CUDNN, ✖ NCCL, ✖ CUDA_RTC, ✖ TENSORRT, ✔ CPU_SSE, ✔ CPU_SSE2, ✔ CPU_SSE3, ✔ CPU_SSE4_1, ✔ CPU_SSE4_2, ✖ CPU_SSE4A, ✔ CPU_AVX, ✖ CPU_AVX2, ✖ OPENMP, ✖ SSE, ✔ F16C, ✖ JEMALLOC, ✔ BLAS_OPEN, ✖ BLAS_ATLAS, ✖ BLAS_MKL, ✖ BLAS_APPLE, ✔ LAPACK, ✖ MKLDNN, ✔ OPENCV, ✖ CAFFE, ✖ PROFILER, ✖ DIST_KVSTORE, ✖ CXX14, ✔ SIGNAL_HANDLER, ✔ DEBUG]

In [3]: fs['CUDA'].enabled
Out[3]: False

In [4]: fs.is_enabled('CPU_SSE')
Out[4]: True

In [5]: fs.is_enabled('CUDA')
Out[5]: False

In [6]:
```


```eval_rst
.. autosummary::
    :nosignatures:

    LibFeature
    libinfo_features
```

## API Reference

<script type="text/javascript" src='../../../_static/js/auto_module_index.js'></script>

```eval_rst
.. automodule:: mxnet.runtime
    :members:
```

<script>auto_index("api-reference");</script>
