MXNet Change Log
================
## 0.11.0
### - Major Features
  - [Apple Core ML model converter](https://github.com/apache/incubator-mxnet/blob/master/tools/coreml/README.md)
  - Support for [Keras v1.2.2](https://github.com/dmlc/keras/wiki/Installation)
  - [Gluon Interface(experimental)](https://github.com/apache/incubator-mxnet/blob/v0.11.0/docs/api/python/gluon.md)
  - Updated [LICENSE and NOTICE](https://github.com/apache/incubator-mxnet/pull/7565) files.
  - For more information see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/v0.11.0+Release+Notes)  
### - API Changes
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
### - Performance Improvements
  - ImageRecordIter now stores data in pinned memory to improve GPU memcopy speed.
### - Bugfixes
  - Cython interface is fixed. `make cython` and `python setup.py install --with-cython` should install the cython interface and reduce overhead in applications that use imperative/bucketing.
  - Fixed various bugs in Faster-RCNN example: https://github.com/dmlc/mxnet/pull/6486
  - Fixed various bugs in SSD example.
  - Fixed `out` argument not working for `zeros`, `ones`, `full`, etc.
  - `expand_dims` now supports backward shape inference.
  - Fixed a bug in rnn. BucketingSentenceIter that causes incorrect layout handling on multi-GPU.
  - Fixed context mismatch when loading optimizer states.
  - Fixed a bug in ReLU activation when using MKL.
  - Fixed a few race conditions that causes crashes on shutdown.
  - Fixed [image-classification example code](https://github.com/apache/incubator-mxnet/pull/7545).
### - Refactors
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
