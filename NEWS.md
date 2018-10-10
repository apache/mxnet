MXNet Change Log
================
## 1.3.0

### New Features - Gluon RNN layers are now HybridBlocks
- In this release, Gluon RNN layers such as `gluon.rnn.RNN`, `gluon.rnn.LSTM`, `gluon.rnn.GRU` becomes `HybridBlock`s as part of [gluon.rnn improvements project](https://github.com/apache/incubator-mxnet/projects/11) (#11482).
- This is the result of newly available fused RNN operators added for CPU: LSTM([#10104](https://github.com/apache/incubator-mxnet/pull/10104)), vanilla RNN([#11399](https://github.com/apache/incubator-mxnet/pull/11399)), GRU([#10311](https://github.com/apache/incubator-mxnet/pull/10311))
- Now many dynamic networks that are based on Gluon RNN layers can now be completely hybridized, exported, and used in the inference APIs in other language bindings such as R, Scala, etc.

### MKL-DNN improvements
- Introducing more functionality support for MKL-DNN as follows:
  - Added support for more activation functions like, "sigmoid", "tanh", "softrelu". ([#10336](https://github.com/apache/incubator-mxnet/pull/10336))
  - Added Debugging functionality: Result check ([#12069](https://github.com/apache/incubator-mxnet/pull/12069)) and Backend switch ([#12058](https://github.com/apache/incubator-mxnet/pull/12058)).

### New Features - Gluon Model Zoo Pre-trained Models
- Gluon Vision Model Zoo now provides MobileNetV2 pre-trained models (#10879) in addition to
  AlexNet, DenseNet, Inception V3, MobileNetV1, ResNet V1 and V2, SqueezeNet 1.0 and 1.1, and VGG
  pretrained models.
- Updated pre-trained models provide state-of-the-art performance on all resnetv1, resnetv2, and vgg16, vgg19, vgg16_bn, vgg19_bn models (#11327 #11860 #11830).

### New Features - Clojure package (experimental)
- MXNet now supports the Clojure programming language. The MXNet Clojure package brings flexible and efficient GPU computing and state-of-art deep learning to Clojure. It enables you to write seamless tensor/matrix computation with multiple GPUs in Clojure. It also lets you construct and customize the state-of-art deep learning models in Clojure, and apply them to tasks, such as image classification and data science challenges.([#11205](https://github.com/apache/incubator-mxnet/pull/11205))
- Checkout examples and API documentation [here](http://mxnet.incubator.apache.org/api/clojure/index.html).

### New Features - Synchronized Cross-GPU Batch Norm (experimental)
- Gluon now supports Synchronized Batch Normalization (#11502).
- This enables stable training on large-scale networks with high memory consumption such as FCN for image segmentation.

### New Features - Sparse Tensor Support for Gluon (experimental)
- Sparse gradient support is added to `gluon.nn.Embedding`. Set `sparse_grad=True` to enable when constructing the Embedding block. ([#10924](https://github.com/apache/incubator-mxnet/pull/10924))
- Gluon Parameter now supports "row_sparse" storage type, which reduces communication cost and memory consumption for multi-GPU training for large models. `gluon.contrib.nn.SparseEmbedding` is an example empowered by this. ([#11001](https://github.com/apache/incubator-mxnet/pull/11001), [#11429](https://github.com/apache/incubator-mxnet/pull/11429))
- Gluon HybridBlock now supports hybridization with sparse operators ([#11306](https://github.com/apache/incubator-mxnet/pull/11306)).

### New Features - Control flow operators (experimental)
- This is the first step towards optimizing dynamic neural networks with variable computation graphs, by adding symbolic and imperative control flow operators. [Proposal](https://cwiki.apache.org/confluence/display/MXNET/Optimize+dynamic+neural+network+models+with+control+flow+operators).
- New operators introduced: foreach([#11531](https://github.com/apache/incubator-mxnet/pull/11531)), while_loop([#11566](https://github.com/apache/incubator-mxnet/pull/11566)), cond([#11760](https://github.com/apache/incubator-mxnet/pull/11760)).

### New Features - Scala API Improvements (experimental)
- Improvements to MXNet Scala API usability([#10660](https://github.com/apache/incubator-mxnet/pull/10660), [#10787](https://github.com/apache/incubator-mxnet/pull/10787), [#10991](https://github.com/apache/incubator-mxnet/pull/10991))
- Symbol.api and NDArray.api would bring new set of functions that have complete definition for all arguments.
- Please see this [Type safe API design document](https://cwiki.apache.org/confluence/display/MXNET/Scala+Type-safe+API+Design+Doc) for more details.

### New Features - Rounding GPU Memory Pool for dynamic networks with variable-length inputs and outputs (experimental)
- MXNet now supports a new memory pool type for GPU memory (#11041).
- Unlike the default memory pool requires exact size match to reuse released memory chunks, this new memory pool uses exponential-linear rounding so that similar sized memory chunks can all be reused, which is more suitable for all the workloads with dynamic-shape inputs and outputs. Set environment variable `MXNET_GPU_MEM_POOL_TYPE=Round` to enable.

### New Features - Topology-aware AllReduce (experimental)
- This features uses trees to perform the Reduce and Broadcast. It uses the idea of minimum spanning trees to do a binary tree Reduce communication pattern to improve it. This topology aware approach reduces the existing limitations for single machine communication shown by mehods like parameter server and NCCL ring reduction. It is an experimental feature ([#11591](https://github.com/apache/incubator-mxnet/pull/11591)).
- Paper followed for implementation: [Optimal message scheduling for aggregation](https://www.sysml.cc/doc/178.pdf).
- Set environment variable `MXNET_KVSTORE_USETREE=1` to enable.

### New Features - Export MXNet models to ONNX format (experimental)
- With this feature, now MXNet models can be exported to ONNX format([#11213](https://github.com/apache/incubator-mxnet/pull/11213)). Currently, MXNet supports ONNX v1.2.1. [API documentation](http://mxnet.incubator.apache.org/api/python/contrib/onnx.html).
- Checkout this [tutorial](http://mxnet.incubator.apache.org/tutorials/onnx/export_mxnet_to_onnx.html) which shows how to use MXNet to ONNX exporter APIs. ONNX protobuf so that those models can be imported in other frameworks for inference.

### New Features - TensorRT Runtime Integration (experimental)
- [TensorRT](https://developer.nvidia.com/tensorrt) provides significant acceleration of model inference on NVIDIA GPUs compared to running the full graph in MxNet using unfused GPU operators. In addition to faster fp32 inference, TensorRT optimizes fp16 inference, and is capable of int8 inference (provided the quantization steps are performed). Besides increasing throughput, TensorRT significantly reduces inference latency, especially for small batches.
- This feature in MXNet now introduces runtime integration of TensorRT into MXNet, in order to accelerate inference.([#11325](https://github.com/apache/incubator-mxnet/pull/11325))
- Currently, its in contrib package.

### New Examples - Scala
- Refurnished Scala Examples with improved API, documentation and CI test coverage. ([#11753](https://github.com/apache/incubator-mxnet/pull/11753), [#11621](https://github.com/apache/incubator-mxnet/pull/11621) )
- Now all Scala examples have:
  - No bugs block in the middle
  - Good Readme to start with
  - with Type-safe API usage inside
  - monitored in CI in each PR runs

### Maintenance - Flaky Tests improvement effort
- Fixed 130 flaky tests on CI. Tracked progress of the project [here](https://github.com/apache/incubator-mxnet/projects/9).
- Add flakiness checker (#11572)

### Maintenance - MXNet Model Backwards Compatibility Checker
- This tool ([#11626](https://github.com/apache/incubator-mxnet/pull/11626)) helps in ensuring consistency and sanity while performing inference on the latest version of MXNet using models trained on older versions of MXNet.
- This tool will help in detecting issues earlier in the development cycle which break backwards compatibility on MXNet and would contribute towards ensuring a healthy and stable release of MXNet.

### Maintenance - Integrated testing for "the Straight Dope"
- ["Deep Learning - The Straight Dope"](http://gluon.mxnet.io) is a deep learning book based on Apache MXNet Gluon that are contributed by many Gluon users.
- Now the testing of this book is integrated in the nightly tests.

### Bug-fixes
- Fix gperftools/jemalloc and lapack warning bug. (#11110)
- Fix mkldnn performance regression + improve test logging (#11262)
- Fix row_sparse_param.save() (#11266)
- Fix trainer init_kvstore (#11266)
- Fix axis Bug in MKLDNN Softmax (#11335)
- Fix 'AttributeError: '_thread._local' object has no attribute 'value'' on distributed processing applications (#11332)
- Fix recordfile dataset with multi worker (#11370)
- Manually check node existence in CachedOp (#11545)
- Javadoc fix (#11239)
- Fix bugs in MKLDNN operators to handle the kAddTo request (#11129)
- Fix InferStorage for sparse fallback in FullyConnected (#11498)
- Fix batchnorm problem with sparse matrices when fix_gamma=True (#11656)
- Fix rnn layer save (#11776)
- Fix BucketSentenceIter bug related to #11430 (#11580)
- Fix for _backward_softsign activation (#11827)
- Fix a bug in CachedOp. (#11675)
- Fix quantization divide by zero errors (#11833)
- Refactor R optimizers to fix memory leak (#11374)
- Avoid use of troublesome cudnnFind() results when grad_req='add' (#11338)
- Fix shared memory with gluon dataloader, add option pin_memory (#11908)
- Fix quantized graph pass bug (#11937)
- Fix MXPredReshape in the c_predict_api (#11493)
- Fix the topk regression issue (#12197)
- Fix image-classification example and add missing optimizers w/ momentum support (#11826)

### Performance Improvements
- Added static allocation and static shape for HybridBloc gluon (#11320)
- Fix RecordIO augmentation speed (#11474)
- Improve sparse pull performance for gluon trainer (#11429)
- CTC operator performance improvement from HawkAaron/MXNet-CTC (#11834)
- Improve performance of broadcast ops backward pass (#11252)
- Improved numerical stability as a result of using stable L2 norm (#11573)
- Accelerate the performance of topk for GPU and CPU side (#12085 #10997 ; This changes the behavior of topk when nan values occur in the input) 
- Support for dot(dns, csr) = dns and dot(dns, csr.T) = dns on CPU ([#11113](https://github.com/apache/incubator-mxnet/pull/11113))
- Performance improvement for Batch Dot on CPU from mshadow ([mshadow PR#342](https://github.com/dmlc/mshadow/pull/342))

### API Changes
- Allow Scala users to specify data/label names for NDArrayIter (#11256)
- Allow user to define unknown token symbol to rnn encode_sentences() (#10461)
- Added count_include_pad argument for Avg Pooling (#11021)
- Add standard ResNet data augmentation for ImageRecordIter (#11027)
- Add seed_aug parameter for ImageRecordIter to fix random seed for default augmentation (#11247)
- Add support for accepting MXNet NDArrays in ColorNormalizeAug (#11606)
- Enhancement of take operator (#11326)
- Add temperature parameter in Softmax operator (#11466)
- Add support for 1D inputs in leaky relu (#11850)
- Add verify_ssl option to gluon.utils.download (#11546)

### Other features
- Added ccache reporting to CI (#11322)
- Restructure dockcross dockerfiles to fix caching (#11302)
- Added tests for MKLDNN backward operators  (#11232)
- Add elemwise_add/sub between rsp and rsp on GPU (#11179)
- Add clip_global_norm(row_sparse_grad) (#11266)
- Add subgraph storage type inference to CachedOp  (#11306)
- Enable support for dense weight and sparse grad Adagrad updates (#11355)
- Added Histogram Operator (#10931)
- Added Matthew's Correlation Coefficient to metrics (#10524)
- Added support for add_n(dense, csr, dense) = dense on CPU & GPU (#11330)
- Added support for add_n(any combination longer than 4 with at least one dense storage) = dense on CPU & GPU (#11330)
- L1 Normalization (#11229)
- Add support for int64 data type in CSVIter (#11446)
- Add test for new int64 type in CSVIter (#11499)
- Add sample ratio for ROI Align (#11145)
- Shape and Size Operator (#10889)
- Add HybidSequentialRNNCell, which can be nested in HybridBlock (#11003)
- Support for a bunch of unary functions for csr matrices (#11559)
- Added NDArrayCollector to dispose intermediate allocated NDArrays automatically (#11751)
- Added the diag() operator (#11643)
- Added broadcast_like operator (#11820)
- Allow Partial shape infer for Slice (#11406)
- Added support to profile kvstore server during distributed training  (#11215)
- Add function for GPU Memory Query to C API (#12083)
- Generalized reshape_like operator to be more flexible (#11928)
- Add support for selu activation function (#12059)
- Add support for accepting NDArray as input to Module predict API (#12166)
- Add DataDesc type for the Scala Package (#11844)

### Usability Improvements
- Added NDArray auto-collector for Scala (#11751, #12232)
- Added docs for mx.initializer.Constant (#10637)
- Added build from souce instructions on windows (#11276)
- Added a tutorial explaining how to use the profiler (#11274)
- Added two tutorials on Learning Rate Schedules (#11296)
- Added a tutorial for mixed precision training with float16 (#10391)
- Create CPP test for concat MKLDNN operator (#11371)
- Update large word language model example (#11405)
- MNIST Examples for Scala new API (#11250)
- Updated installation info to have latest packages and more clarity (#11503)
- GAN MNIST Examples for Scala new API (#11547)
- Added Learning Rate Finder tutorial (#11304)
- Fix Installation instructions for R bindings on Linux systems. (#11590)
- Integration Test for Scala (#11596)
- Documentation enhancement for optimizers (#11657)
- Update rcnn example (#11373)
- Gluon ModelZoo, Gluon examples for Perl APIs (#11642)
- Fix R installation in CI (#11761,#11755, #11768, #11805, #11954, #11976)
- CNN Examples for Scala new API (#11292)
- Custom Operator Example for Scala (#11401)
- Added detailed doc about global pool layers in Gluon (#11832)
- Updated MultiTask example to use new infer api (#11605)
- Added logistic regression tutorial (#11651)
- Added Support for integer type in ImageIter (#11864)
- Added depth_to_space and space_to_depth operators (#11587)
- Increased operator support for ONNX to MXNet importer (#11856)
- Add linux and macos MKLDNN Building Instruction (#11049)
- Add download utility for Scala APIs (#11866)
- Improving documentation and error messages for Async distributed training with Gluon (#11910)
- Added NeuralStyle Example for Scala (#11621)

For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+%28incubating%29+1.3.0+Release+Notes)

## 1.2.0
### New Features - Added Scala Inference APIs
- Implemented new [Scala Inference APIs](https://cwiki.apache.org/confluence/display/MXNET/MXNetScalaInferenceAPI) which offer an easy-to-use, Scala Idiomatic and thread-safe high level APIs for performing predictions with deep learning models trained with MXNet (#9678). Implemented a new ImageClassifier class which provides APIs for classification tasks on a Java BufferedImage using a pre-trained model you provide (#10054). Implemented a new ObjectDetector class which provides APIs for object and boundary detections on a Java BufferedImage using a pre-trained model you provide (#10229).

### New Features - Added a Module to Import ONNX models into MXNet
- Implemented a new ONNX module in MXNet which offers an easy to use API to import ONNX models into MXNet's symbolic interface (#9963). Checkout the [example](https://github.com/apache/incubator-mxnet/blob/master/example/onnx/super_resolution.py) on how you could use this [API](https://cwiki.apache.org/confluence/display/MXNET/ONNX-MXNet+API+Design) to import ONNX models and perform inference on MXNet. Currently, the ONNX-MXNet Import module is still experimental. Please use it with caution.

### New Features - Added Support for Model Quantization with Calibration
- Implemented model quantization by adopting the [TensorFlow approach](https://www.tensorflow.org/performance/quantization) with calibration by borrowing the idea from Nvidia's [TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf). The focus of this work is on keeping quantized models (ConvNets for now) inference accuracy loss under control when compared to their corresponding FP32 models. Please see the [example](https://github.com/apache/incubator-mxnet/tree/master/example/quantization) on how to quantize a FP32 model with or without calibration (#9552). Currently, the Quantization support is still experimental. Please use it with caution.

### New Features - MKL-DNN Integration
- MXNet now integrates with Intel MKL-DNN to accelerate neural network operators: Convolution, Deconvolution, FullyConnected, Pooling, Batch Normalization, Activation, LRN, Softmax, as well as some common operators: sum and concat (#9677). This integration allows NDArray to contain data with MKL-DNN layouts and reduces data layout conversion to get the maximal performance from MKL-DNN. Currently, the MKL-DNN integration is still experimental. Please use it with caution.

### New Features - Added Exception Handling Support for Operators
- Implemented [Exception Handling Support for Operators](https://cwiki.apache.org/confluence/display/MXNET/Improved+exception+handling+in+MXNet) in MXNet. MXNet now transports backend C++ exceptions to the different language front-ends and prevents crashes when exceptions are thrown during operator execution (#9681).

### New Features - Enhanced FP16 support
- Added support for distributed mixed precision training with FP16. It supports storing of master copy of weights in float32 with the multi_precision mode of optimizers (#10183). Improved speed of float16 operations on x86 CPU by 8 times through F16C instruction set. Added support for more operators to work with FP16 inputs (#10125, #10078, #10169). Added a tutorial on using mixed precision with FP16 (#10391).

### New Features - Added Profiling Enhancements
- Enhanced built-in profiler to support native Intel:registered: VTune:tm: Amplifier objects such as Task, Frame, Event, Counter and Marker from both C++ and Python -- which is also visible in the Chrome tracing view(#8972). Added Runtime tracking of symbolic and imperative operators as well as memory and API calls. Added Tracking and dumping of aggregate profiling data. Profiler also no longer affects runtime performance when not in use. 

### Breaking Changes
- Changed Namespace for MXNet scala from `ml.dmlc.mxnet` to `org.apache.mxnet` (#10284).
- Changed API for the Pooling operator from `mxnet.symbol.Pooling(data=None, global_pool=_Null, cudnn_off=_Null, kernel=_Null, pool_type=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, name=None, attr=None, out=None, **kwargs)` to  `mxnet.symbol.Pooling(data=None,  kernel=_Null, pool_type=_Null, global_pool=_Null, cudnn_off=_Null, pooling_convention=_Null, stride=_Null, pad=_Null, name=None, attr=None, out=None, **kwargs)`. This is a breaking change when kwargs are not provided since the new api expects the arguments starting from `global_pool` at the fourth position instead of the second position. (#10000).

### Bug Fixes
- Fixed tests - Flakiness/Bugs - (#9598, #9951, #10259, #10197, #10136, #10422). Please see: [Tests Improvement Project](https://github.com/apache/incubator-mxnet/projects/9)
- Fixed `cudnn_conv` and `cudnn_deconv` deadlock (#10392).
- Fixed a race condition in `io.LibSVMIter` when batch size is large (#10124).
- Fixed a race condition in converting data layouts in MKL-DNN (#9862).
- Fixed MKL-DNN sigmoid/softrelu issue (#10336).
- Fixed incorrect indices generated by device row sparse pull (#9887).
- Fixed cast storage support for same stypes (#10400).
- Fixed uncaught exception for bucketing module when symbol name not specified (#10094).
- Fixed regression output layers (#9848).
- Fixed crash with `mx.nd.ones` (#10014).
- Fixed `sample_multinomial` crash when `get_prob=True` (#10413).
- Fixed buggy type inference in correlation (#10135).
- Fixed race condition for `CPUSharedStorageManager->Free` and launched workers at iter init stage to avoid frequent relaunch (#10096).
- Fixed DLTensor Conversion for int64 (#10083).
- Fixed issues where hex symbols of the profiler were not being recognized by chrome tracing tool(#9932)
- Fixed crash when profiler was not enabled (#10306)
- Fixed ndarray assignment issues (#10022, #9981, #10468).
- Fixed incorrect indices generated by device row sparse pull (#9887).
- Fixed `print_summary` bug in visualization module (#9492).
- Fixed shape mismatch in accuracy metrics (#10446).
- Fixed random samplers from uniform and random distributions in R bindings (#10450).
- Fixed a bug that was causing training metrics to be printed as NaN sometimes (#10437).
- Fixed a crash with non positive reps for tile ops (#10417).

### Performance Improvements 
- On average, after the MKL-DNN change, the inference speed of MXNet + MKLDNN outperforms MXNet + OpenBLAS by a factor of 32, outperforms MXNet + MKLML by 82% and outperforms MXNet + MKLML with the experimental flag by 8%. The experiments were run for the image classifcation example, for different networks and different batch sizes.
- Improved sparse SGD, sparse AdaGrad and sparse Adam optimizer speed on GPU by 30x (#9561, #10312, #10293, #10062).
- Improved `sparse.retain` performance on CPU by 2.5x (#9722)
- Replaced `std::swap_ranges` with memcpy (#10351)
- Implemented DepthwiseConv2dBackwardFilterKernel which is over 5x faster (#10098)
- Implemented CPU LSTM Inference (#9977)
- Added Layer Normalization in C++ (#10029)
- Optimized Performance for rtc (#10018)
- Improved CPU performance of  ROIpooling operator by using OpenMP (#9958)
- Accelerated the calculation of F1 (#9833)

### API Changes
- `Block.save_params` now match parameters according to model structure instead of names to avoid prefix mismatching problems during saving and loading (#10511).
- Added an optional argument `ctx` to `mx.random.seed`. Seeding with `ctx` option produces random number sequence independent of device id. (#10367).
- Added copy flag for astype (#10347).
- Added context parameter to Scala Infer API - ImageClassifier and ObjectDetector (#10252).
- Added axes support for dropout in gluon (#10032).
- Added default `ctx` to cpu for `gluon.Block.load_params` (#10160).
- Added support for variable sequence length in gluon.RecurrentCell (#9934).
- Added convenience fluent method for squeeze op (#9734).
- Made `array.reshape` compatible with numpy (#9790).
- Added axis support and gradient for L2norm (#9740).

### Sparse Support
- Added support for multi-GPU training with `row_sparse` weights using `device` KVStore (#9987).
- Added `Module.prepare` API for multi-GPU and multi-machine training with row_sparse weight (#10285).
- Added `deterministic` option for `contrib.SparseEmbedding` operator (#9846).
- Added `sparse.broadcast_mul` and `sparse.broadcast_div` with CSRNDArray and 1-D dense NDArray on CPU (#10208).
- Added sparse support for Custom Operator (#10374).
- Added Sparse feature for Perl (#9988).
- Added `force_deterministic` option for sparse embedding (#9882).
- Added `sparse.where` with condition being csr ndarray (#9481).

### Deprecations
- Deprecated `profiler_set_state` (#10156).

### Other Features
- Added constant parameter for gluon (#9893).
- Added `contrib.rand.zipfian` (#9747).
- Added Gluon PreLU, ELU, SELU, Swish activation layers for Gluon (#9662)
- Added Squeeze Op (#9700).
- Added multi-proposal operator (CPU version) and fixed bug in multi-proposal operator (GPU version) (#9939).
- Added in Large-Batch SGD with a warmup, and a LARS startegy (#8918).
- Added Language Modelling datasets and Sampler (#9514).
- Added instance norm and reflection padding to Gluon (#7938).
- Added micro-averaging strategy for F1 metric (#9777).
- Added Softsign Activation Function (#9851).
- Added eye operator, for default storage type (#9770).
- Added TVM bridge support to JIT NDArray Function by TVM (#9880).
- Added float16 support for correlation operator and L2Normalization operator (#10125, #10078).
- Added random shuffle implementation for NDArray (#10048).
- Added load from buffer functions for CPP package (#10261).

### Usability Improvements
- Added embedding learning example for Gluon (#9165).
- Added tutorial on how to use data augmenters (#10055).
- Added tutorial for Data Augmentation with Masks (#10178).
- Added LSTNet example (#9512).
- Added MobileNetV2 example (#9614).
- Added tutorial for Gluon Datasets and DataLoaders (#10251).
- Added Language model with Google's billion words dataset (#10025).
- Added example for custom operator using RTC (#9870).
- Improved image classification examples (#9799, #9633).
- Added reshape predictor function to c_predict_api (#9984).
- Added guide for implementing sparse ops (#10081).
- Added naming tutorial for gluon blocks and parameters (#10511).

### Known Issues
- MXNet crash when built with `USE_GPERFTOOLS = 1` (#8968).
- [DevGuide.md](https://github.com/google/googletest/blob/ec44c6c1675c25b9827aacd08c02433cccde7780/googlemock/docs/DevGuide.md) in the 3rdparty submodule googletest licensed under CC-BY-2.5.
- Incompatibility in the behavior of MXNet Convolution operator for certain unsupported use cases: Raises an exception when MKLDNN is enabled, fails silently when it is not.
- MXNet convolution generates wrong results for 1-element strides (#10689).
- [Tutorial on fine-tuning an ONNX model](https://github.com/apache/incubator-mxnet/blob/v1.2.0/docs/tutorials/onnx/fine_tuning_gluon.md) fails when using cpu context.
- CMake build ignores the `USE_MKLDNN` flag and doesn't build with MKLDNN support even with `-DUSE_MKLDNN=1`. To workaround the issue please see: #10801.
- Linking the dmlc-core library fails with CMake build when building with `USE_OPENMP=OFF`. To workaround the issue, please use the updated CMakeLists in dmlc-core unit tests directory: https://github.com/dmlc/dmlc-core/pull/396. You can also workaround the issue by using make instead of cmake when building with `USE_OPENMP=OFF`.

For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/%5BWIP%5D+Apache+MXNet+%28incubating%29+1.2.0+Release+Notes)

## 1.1.0
### Usability Improvements
- Improved the usability of examples and tutorials
### Bug-fixes
- Fixed I/O multiprocessing for too many open file handles (#8904), race condition (#8995), deadlock (#9126).
- Fixed image IO integration with OpenCV 3.3 (#8757).
- Fixed Gluon block printing (#8956).
- Fixed float16 argmax when there is negative input. (#9149)
- Fixed random number generator to ensure sufficient randomness. (#9119, #9256, #9300)
- Fixed custom op multi-GPU scaling (#9283)
- Fixed gradient of gather_nd when duplicate entries exist in index. (#9200)
- Fixed overriden contexts in Module `group2ctx` option when using multiple contexts (#8867)
- Fixed `swap_axes` operator with "add_to" gradient req (#9541)
### New Features
- Added experimental API in `contrib.text` for building vocabulary, and loading pre-trained word embeddings, with built-in support for 307 GloVe and FastText pre-trained embeddings. (#8763)
- Added experimental structural blocks in `gluon.contrib`: `Concurrent`, `HybridConcurrent`, `Identity`. (#9427)
- Added `sparse.dot(dense, csr)` operator (#8938)
- Added `Khatri-Rao` operator (#7781)
- Added `FTML` and `Signum` optimizer (#9220, #9262)
- Added `ENABLE_CUDA_RTC` build option (#9428)
### API Changes
- Added zero gradients to rounding operators including `rint`, `ceil`, `floor`, `trunc`, and `fix` (#9040)
- Added `use_global_stats` in `nn.BatchNorm` (#9420)
- Added `axis` argument to `SequenceLast`, `SequenceMask` and `SequenceReverse` operators (#9306)
- Added `lazy_update` option for standard `SGD` & `Adam` optimizer with `row_sparse` gradients (#9468, #9189)
- Added `select` option in `Block.collect_params` to support regex (#9348)
- Added support for (one-to-one and sequence-to-one) inference on explicit unrolled RNN models in R (#9022) 
### Deprecations
- The Scala API name space is still called `ml.dmlc`. The name space is likely be changed in a future release to `org.apache` and might brake existing applications and scripts (#9579, #9324)
### Performance Improvements
- Improved GPU inference speed by 20% when batch size is 1 (#9055)
- Improved `SequenceLast` operator speed (#9306)
- Added multithreading for the class of broadcast_reduce operators on CPU (#9444)
- Improved batching for GEMM/TRSM operators with large matrices on GPU (#8846)
### Known Issues
- "Predict with pre-trained models" tutorial is broken
- "example/numpy-ops/ndarray_softmax.py" is broken

For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/display/MXNET/Apache+MXNet+%28incubating%29+1.1.0+Release+Notes)


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
