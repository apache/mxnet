# Run-Time Feature detection / Library info

```eval_rst
.. currentmodule:: mxnet.runtime
```

## Overview

The libinfo functionality allows to check for compile-time features supported by the library.

### Example usage

```python
In [1]: import mxnet as mx

In [2]: import mxnet.runtime

In [3]: mx.runtime.libinfo_features()
Out[3]:
[✔ CUDA,
 ✔ CUDNN,
 ✔ NCCL,
 ✔ CUDA_RTC,
 ✔ TENSORRT,
 ✖ CPU_SSE,
 ✖ CPU_SSE2,
 ✖ CPU_SSE3,
 ✖ CPU_SSE4_1,
 ✖ CPU_SSE4_2,
 ✔ CPU_SSE4A,
 ✖ CPU_AVX,
 ✔ CPU_AVX2,
 ✔ OPENMP,
 ✔ SSE,
 ✖ F16C,
 ✔ JEMALLOC,
 ✖ BLAS_OPEN,
 ✔ BLAS_ATLAS,
 ✔ BLAS_MKL,
 ✔ BLAS_APPLE,
 ✔ LAPACK,
 ✔ MKLDNN,
 ✖ OPENCV,
 ✔ CAFFE,
 ✔ PROFILER,
 ✔ DIST_KVSTORE,
 ✔ CXX14,
 ✔ SIGNAL_HANDLER,
 ✖ DEBUG]
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
