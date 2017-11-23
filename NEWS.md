MXNet Change Log
================
## 1.0.0
### Performance
  - Enhanced the performance of `sparse.dot` operator.
  - MXNet now automatically set OpenMP to use all available CPU cores to maximize CPU utilization when `NUM_OMP_THREADS` is not set.
  - Unary and binary operators now avoid using OpenMP on small arrays if using OpenMP actually hurts performance due to multithreading overhead.
  - Significantly improved performance of `broadcast_add`, `broadcast_mul`, etc on CPU.
  - Added bulk execution to imperative mode. You can control segment size with `mxnet.engine.bulk`. As a result, the speed of Gluon in hybrid mode is improved, especially on small networks and multiple GPUs.
  - Improved speed for `ctypes` invocation from Python frontend.
### New Features - Gradient Compression [Experimental]
  - Speed up multi-GPU and distributed training by compressing communication of gradients. This is especially effective when training networks with large fully-connected layers. In Gluon this can be activated with `compression_params` in Trainer.
### New Features - Support of NVIDIA Collective Communication Library (NCCL) [Experimental]
  - Use `kvstore=’nccl’` for (in some cases) faster training on multiple GPUs.
  - Significantly faster than kvstore=’device’ when batch size is small.
  - It is recommended to set environment variable `NCCL_LAUNCH_MODE` to `PARALLEL` when using NCCL version 2.1 or newer.
### New Features - Advanced Indexing [General Availability]
  - NDArray now supports advanced indexing (both slice and assign) as specified by the numpy standard: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#combining-advanced-and-basic-indexing with the following restrictions:
    - if key is a list type, only a list of integers is supported, e.g. `key=[1, 2]` is supported, while not for `key=[[1, 2]]`.
    - Ellipsis (...) and np.newaxis are not supported.
    - `Boolean` array indexing is not supported.
### New Features - Gluon [General Availability]
  - Performance optimizations discussed above.
  - Added support for loading data in parallel with multiple processes to `gluon.data.DataLoader`. The number of workers can be set with `num_worker`. Does not support windows yet.
  - Added Block.cast to support networks with different data types, e.g. `float16`.
  - Added Lambda block for wrapping a user defined function as a block.
  - Generalized `gluon.data.ArrayDataset` to support arbitrary number of arrays.
### New Features - ARM / Raspberry Pi support [Experimental]
  - MXNet now compiles and runs on ARMv6, ARMv7, ARMv64 including Raspberry Pi devices. See https://github.com/apache/incubator-mxnet/tree/master/docker_multiarch for more information.
### New Features - NVIDIA Jetson support [Experimental]
  - MXNet now compiles and runs on NVIDIA Jetson TX2 boards with GPU acceleration.
  - You can install the python MXNet package on a Jetson board by running - `$ pip install mxnet-jetson-tx2`.
### New Features - Sparse Tensor Support [General Availability]
  - Added more sparse operators: `contrib.SparseEmbedding`, `sparse.sum` and `sparse.mean`. 
  - Added `asscipy()` for easier conversion to scipy.
  - Added `check_format()` for sparse ndarrays to check if the array format is valid.
### Bug-fixes  
  - Fixed a[-1] indexing doesn't work on `NDArray`.
  - Fixed `expand_dims` if axis < 0.
  - Fixed a bug that causes topk to produce incorrect result on large arrays.
  - Improved numerical precision of unary and binary operators for `float64` data.
  - Fixed derivatives of log2 and log10. They used to be the same with log.
  - Fixed a bug that causes MXNet to hang after fork. Note that you still cannot use GPU in child processes after fork due to limitations of CUDA.
  - Fixed a bug that causes `CustomOp` to fail when using auxiliary states.
  - Fixed a security bug that is causing MXNet to listen on all available interfaces when running training in distributed mode.
### Doc Updates
  - Added a security best practices document under FAQ section.
  - Fixed License Headers including restoring copyright attributions.
  - Documentation updates. 
  - Links for viewing source.
 
 For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+%28incubating%29+1.0+Release+Notes)


## 0.12.1
### Bug-fixes
  - Added GPU support for the `syevd` operator which ensures that there is GPU support for all linalg-operators.
  - Bugfix for `syevd` on CPU such that it works for `float32`.
  - Fixed API call when `OMP_NUM_THREADS` environment variable is set. 
  - Fixed `MakeNonlossGradNode` bug.
  - Fixed bug related to passing `dtype` to `array()`. 
  - Fixed some minor bugs for sparse distributed training.
  - Fixed a bug on `Slice` accessing uninitialized memory in `param.begin` in the file `matrix_op-inl.h`. 
  - Fixed `gluon.data.RecordFileDataset`.
  - Fixed a bug that caused `autograd` to crash on some networks.
  
  
## 0.12.0
### Performance
  - Added full support for NVIDIA Volta GPU Architecture and CUDA 9. Training CNNs is up to 3.5x faster than Pascal when using float16 precision.
  - Enabled JIT compilation. Autograd and Gluon hybridize now use less memory and has faster speed. Performance is almost the same with old symbolic style code.
  - Improved ImageRecordIO image loading performance and added indexed RecordIO support.
  - Added better openmp thread management to improve CPU performance.
### New Features - Gluon
  - Added enhancements to the Gluon package, a high-level interface designed to be easy to use while keeping most of the flexibility of low level API. Gluon supports both imperative and symbolic programming, making it easy to train complex models imperatively with minimal impact on performance. Neural networks (and other machine learning models) can be defined and trained with `gluon.nn` and `gluon.rnn` packages. 
  - Added new loss functions - `SigmoidBinaryCrossEntropyLoss`, `CTCLoss`, `HuberLoss`, `HingeLoss`, `SquaredHingeLoss`, `LogisticLoss`, `TripletLoss`.
  - `gluon.Trainer` now allows reading and setting learning rate with `trainer.learning_rate` property.
  - Added API `HybridBlock.export` for exporting gluon models to MXNet format.
  - Added `gluon.contrib` package.
    - Convolutional recurrent network cells for RNN, LSTM and GRU.
    - `VariationalDropoutCell`
### New Features - Autograd
  - Added enhancements to `autograd` package, which enables automatic differentiation of NDArray operations.
  - `autograd.Function` allows defining both forward and backward computation for custom operators.
  - Added `mx.autograd.grad` and experimental second order gradient support (most operators don't support second order gradient yet).
  - Autograd now supports cross-device graphs. Use `x.copyto(mx.gpu(i))` and `x.copyto(mx.cpu())` to do computation on multiple devices.
### New Features - Sparse Tensor Support
  - Added support for sparse matrices. 
  - Added limited cpu support for two sparse formats in `Symbol` and `NDArray` - `CSRNDArray` and `RowSparseNDArray`.
  - Added a sparse dot product operator and many element-wise sparse operators.
  - Added a data iterator for sparse data input - `LibSVMIter`.
  - Added three optimizers for sparse gradient updates: `Ftrl`, `SGD` and `Adam`.
  - Added `push` and `row_sparse_pull` with `RowSparseNDArray` in distributed kvstore.
### Other New Features
  - Added limited support for fancy indexing, which allows you to very quickly access and modify complicated subsets of an array's values. `x[idx_arr0, idx_arr1, ..., idx_arrn]` is now supported. Features such as combining and slicing are planned for the next release. Checkout master to get a preview.
  - Random number generators in `mx.nd.random.*` and `mx.sym.random.*` now support both CPU and GPU.
  - `NDArray` and `Symbol` now supports "fluent" methods. You can now use `x.exp()` etc instead of `mx.nd.exp(x)` or `mx.sym.exp(x)`.
  - Added `mx.rtc.CudaModule` for writing and running CUDA kernels from python. 
  - Added `multi_precision` option to optimizer for easier float16 training.
  - Better support for IDE auto-completion. IDEs like PyCharm can now correctly parse mxnet operators.
### API Changes
  - Operators like `mx.sym.linalg_*` and `mx.sym.random_*` are now moved to `mx.sym.linalg.*` and `mx.sym.random.*`. The old names are still available but deprecated.
  - `sample_*` and `random_*` are now merged as `random.*`, which supports both scalar and  `NDArray` distribution parameters.
### Bug-fixes
  - Fixed a bug that causes `argsort` operator to fail on large tensors.
  - Fixed numerical stability issues when summing large tensors.
  - Fixed a bug that causes arange operator to output wrong results for large ranges.
  - Improved numerical precision for unary and binary operators on `float64` inputs.

For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/MXNet+0.12.0+Release+Notes)


## 0.11.0
### Major Features
  - Apple Core ML model converter
  - Support for Keras v1.2.2
  - For more information see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/v0.11.0+Release+Notes)
### API Changes
  - Added `CachedOp`. You can now cache the operators that’s called frequently with the same set of arguments to reduce overhead.
  - Added sample_multinomial for sampling from multinomial distributions.
  - Added `trunc` operator for rounding towards zero.
  - Added linalg_gemm, linalg_potrf, ... operators for lapack support.
  - Added verbose option to Initializer for printing out initialization details.
  - Added DeformableConvolution to contrib from the Deformable Convolutional Networks paper.
  - Added float64 support for dot and batch_dot operator.
  - `allow_extra` is added to Module.set_params to ignore extra parameters.
  - Added `mod` operator for modulo.
  - Added `multi_precision` option to SGD optimizer to improve training with float16. Resnet50 now achieves the same accuracy when trained with float16 and gives 50% speedup on Titan XP.
### Performance Improvements
  - ImageRecordIter now stores data in pinned memory to improve GPU memcopy speed.
### Bugfixes
  - Cython interface is fixed. `make cython` and `python setup.py install --with-cython` should install the cython interface and reduce overhead in applications that use imperative/bucketing.
  - Fixed various bugs in Faster-RCNN example: https://github.com/dmlc/mxnet/pull/6486
  - Fixed various bugs in SSD example.
  - Fixed `out` argument not working for `zeros`, `ones`, `full`, etc.
  - `expand_dims` now supports backward shape inference.
  - Fixed a bug in rnn. BucketingSentenceIter that causes incorrect layout handling on multi-GPU.
  - Fixed context mismatch when loading optimizer states.
  - Fixed a bug in ReLU activation when using MKL.
  - Fixed a few race conditions that causes crashes on shutdown.
### Refactors
  - Refactored TShape/TBlob to use int64 dimensions and DLTensor as internal storage. Getting ready for migration to DLPack. As a result TBlob::dev_mask_ and TBlob::stride_ are removed.


## 0.10.0
- Overhauled documentation for commonly used Python APIs, Installation instructions, Tutorials, HowTos and MXNet Architecture.  
- Updated mxnet.io for improved readability.  
- Pad operator now support reflection padding.  
- Fixed a memory corruption error in threadedengine.  
- Added CTC loss layer to contrib package. See mx.contrib.sym.ctc_loss.  
- Added new sampling operators for several distributions (normal,uniform,gamma,exponential,negative binomial).  
- Added documentation for experimental RNN APIs.
 
## 0.9.3
- Move symbolic API to NNVM @tqchen
  - Most front-end C API are backward  compatible
  - Removed symbolic API in MXNet and relies on NNVM
- New features:
  - MXNet profiler for profiling operator-level executions
  - mxnet.image package for fast image loading and processing
- Change of JSON format
  - param and attr field are merged to attr
  - New code is backward-compatible can load old json format
- OpProperty registration now is deprecated
  - New operators are encouraged to register their property to NNVM op registry attribute
- Known features removed limitations to be fixed
  - Bulk segment execution not yet added.

## v0.8
This is the last release before the NNVM refactor.
- CaffeOp and CaffeIter for interfacing with Caffe by @HrWangChengdu @cjolivier01
- WrapCTC plugin for sequence learning by @xlvector
- Improved Multi-GPU performance by @mli
- CuDNN RNN support by @sbodenstein
- OpenCV plugin for parallel image IO by @piiswrong
- More operators as simple op
    - Simple OP @tqchen
    - element wise op with axis and broadcast @mli @sxjscience
- Cudnn auto tuning for faster convolution by @piiswrong
- More applications
    - Faster RCNN by @precedenceguo


## v0.7
-  0.6 is skipped because there are a lot of improvements since initial release
- More math operators
  - elementwise ops and binary ops
- Attribute support in computation graph
  - Now user can use attributes to give various hints about specific learning rate, allocation plans etc
- MXNet is more memory efficient
  - Support user defined memory optimization with attributes
- Support mobile applications by @antinucleon
- Refreshed update of new documents
- Model parallel training of LSTM by @tqchen
- Simple operator refactor by @tqchen
  - add operator_util.h to enable quick registration of both ndarray and symbolic ops
- Distributed training by @mli
- Support Torch Module by @piiswrong
  - MXNet now can use any of the modules from Torch.
- Support custom native operator by @piiswrong
- Support data types including fp16, fp32, fp64, int32, and uint8 by @piiswrong
- Support monitor for easy printing and debugging by @piiswrong
- Support new module API by @pluskid
  - Module API is a middle level API that can be used in imperative manner like Torch-Module
- Support bucketing API for variable length input by @pluskid
- Support CuDNN v5 by @antinucleon
- More applications
  - Speech recognition by @yzhang87
  - [Neural art](https://github.com/dmlc/mxnet/tree/master/example/neural-style) by @antinucleon
  - [Detection](https://github.com/dmlc/mxnet/tree/master/example/rcnn), RCNN bt @precedenceguo
  - [Segmentation](https://github.com/dmlc/mxnet/tree/master/example/fcn-xs), FCN by @tornadomeet
  - [Face identification](https://github.com/tornadomeet/mxnet-face) by @tornadomeet
  - More on the example

## v0.5 (initial release)
- All basic modules ready
