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

MXNet Change Log
================

## 1.4.0

- [New Features](#new-features)
  * [Java Inference API](#java-inference-api)
  * [Julia API](#julia-api)
  * [Control Flow Operators (experimental)](#control-flow-operators--experimental-)
  * [SVRG Optimization](#svrg-optimization)
  * [Subgraph API (experimental)](#subgraph-api--experimental-)
  * [JVM Memory Management](#jvm-memory-management)
  * [Topology-aware AllReduce (experimental)](#topology-aware-allreduce--experimental-)
  * [MKLDNN backend: Graph optimization and Quantization (experimental)](#mkldnn-backend--graph-optimization-and-quantization--experimental-)
    + [Graph Optimization](#graph-optimization)
    + [Quantization](#quantization)
- [New Operators](#new-operators)
- [Feature improvements](#feature-improvements)
  * [Operator](#operator)
  * [Optimizer](#optimizer)
  * [Sparse](#sparse)
  * [ONNX](#onnx)
  * [MKLDNN](#mkldnn)
  * [Inference](#inference)
  * [Other](#other)
- [Frontend API updates](#frontend-api-updates)
  * [Gluon](#gluon)
  * [Symbol](#symbol)
- [Language API updates](#language-api-updates)
  * [Java](#java)
  * [R](#r)
  * [Scala](#scala)
  * [Clojure](#clojure)
  * [Perl](#perl)
  * [Julia](#julia)
- [Performance benchmarks and improvements](#performance-benchmarks-and-improvements)
- [Bug fixes](#bug-fixes)
- [Licensing updates](#licensing-updates)
- [Improvements](#improvements)
  * [Tutorial](#tutorial)
  * [Example](#example)
  * [Documentation](#documentation)
  * [Website](#website)
  * [MXNet Distributions](#mxnet-distributions)
  * [Installation](#installation)
  * [Build and CI](#build-and-ci)
  * [3rd party](#3rd-party)
    + [TVM:](#tvm-)
    + [CUDNN:](#cudnn-)
    + [Horovod:](#horovod-)
- [Deprications](#deprications)
- [Other](#other-1)
- [How to build MXNet](#how-to-build-mxnet)
- [List of submodules used by Apache MXNet (Incubating) and when they were updated last](#list-of-submodules-used-by-apache-mxnet--incubating--and-when-they-were-updated-last)
### New Features
#### Java Inference API

Model inference is often managed in a production ecosystem using primarily Java/Scala tools and frameworks. This release seeks to alleviate the need for software engineers to write custom MXNet wrappers to fit their production environment.

Inference on a trained model has a couple of common use cases:

  1. Real-time or Online Inference - tasks that require immediate feedback, such as fraud detection
  2. Batch or Offline Inference - tasks that don't require immediate feedback, these are use cases where you have massive amounts of data and want to run inference or pre-compute inference results
Real-time Inference is often performed and deployed on popular web frameworks such as Tomcat, Netty, Jetty, etc., all of which use Java.
Batch Inference is often performed on big data platforms such as Spark using Scala or Java.

With this project, we had the following goals:
* Build a new set of APIs that are Java friendly, compatible with Java 7+, are easy to use for inference.
* Lower the barrier to entry of consuming MXNet for production use cases.

More details can be found at the [Java Inference API document](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Java+Inference+API).

#### Julia API

MXNet.jl is the Julia package of Apache MXNet. MXNet.jl brings flexible and efficient GPU computing and state-of-art deep learning to Julia. Some highlights of features include:

  * Efficient tensor/matrix computation across multiple devices, including multiple CPUs, GPUs and distributed server nodes.
  * Flexible manipulation of symbolic to composite for construction of state-of-the-art deep learning models.

#### Control Flow Operators (experimental)

Today we observe more and more dynamic neural network models, especially in the fields of natural language processing and graph analysis. The dynamics in these models come from multiple sources, including:

  * Models are expressed with control flow, such as conditions and loops;
  * NDArrays in a model may have dynamic shapes, meaning the NDArrays of a model or some of the NDArrays have different shapes for different batches;
  * Models may want to use more dynamic data structures, such as lists or dictionaries.
It's natural to express dynamic models in frameworks with an imperative programming interface (e.g., Gluon, Pytorch, TensorFlow Eager). In this kind of interface, developers can use Python control flows, or NDArrays with any shape at any moment, or use Python lists and dictionaries to store data as they want. The problem of this approach is that it highly dependent on the originating front-end programming language (mainly Python). A model implemented in one language can only run in the same language.

A common use case is that machine learning scientists want to develop their models in Python, whereas engineers who deploy the models usually have to use a different "production" language (e.g., Java or C). Gluon tries to close the gap between the model development and production deployment. Machine learning scientists design and implement their models in Python with the imperative interface, and then Gluon converts the implementations from imperative to symbolic by invoking `hybridize()` for model exporting.

The goal of this project is to enhance Gluon to turn a dynamic neural network into a static computation graph. The dynamic control flows are expressed by control flow operators with Gluon hybridization, and these are exported for deployment.

More information can be found at [Optimize dynamic neural network models with control flow operators](https://cwiki.apache.org/confluence/display/MXNET/Optimize+dynamic+neural+network+models+with+control+flow+operators)

#### SVRG Optimization

SVRG stands for Stochastic Variance Reduced Gradient, which was first introduced in the paper [Accelerating Stochastic Gradient Descent using Predicative Variance Reduction in 2013](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf). It is an optimization technique that complements SGD.

SGD is known for large scale optimization, but it suffers from slow convergence asymptotically due to the inherent variance. SGD approximates the full gradient using a small batch of samples which introduces variance. In order to converge faster, SGD often needs to start with a smaller learning rate.

SVRG remedies the slow convergence problem by keeping a version of the estimated weights that is close to the optimal parameters and maintains the average of the full gradient over the full pass of data. The average of the full gradients of all data is calculated w.r.t to parameters of last mth epochs. It has provable guarantees for strongly convex smooth functions; a detailed proof can be found in section 3 of the [paper](https://papers.nips.cc/paper/4937-accelerating-stochastic-gradient-descent-using-predictive-variance-reduction.pdf). SVRG uses a different update rule than SGD: gradients w.r.t current parameters minus gradients w.r.t parameters from the last mth epoch, plus the average of gradients over all data.

Key Characteristics of SVRG:

  * Explicit variance reduction
  * Ability to use relatively large learning rate compared to SGD, which leads to faster convergence.
More details can be found at [SVRG Optimization in MXNet Python Module](https://cwiki.apache.org/confluence/display/MXNET/Unified+integration+with+external+backend+libraries)

#### Subgraph API (experimental)

MXNet can integrate with many different kinds of backend libraries, including TVM, MKLDNN, TensorRT, Intel nGraph and more. In general, these backends support a limited number of operators, so running computation in a model usually involves an interaction between backend-supported operators and MXNet operators. These backend libraries share some common requirements:

TVM , MKLDNN and nGraph use customized data formats. Interaction between these backends with MXNet requires data format conversion.
TVM, MKLDNN, TensorRT and nGraph fuses operators.
Integration with these backends should happen in the granularity of subgraphs instead of in the granularity of operators. To fuse operators, it's obvious that we need to divide a graph into subgraphs so that the operators in a subgraph can be fused into a single operator. To handle customized data formats, we should partition a computation graph into subgraphs as well. Each subgraph contains only TVM, MKLDNN or nGraph operators. In this way, MXNet converts data formats only when entering such a subgraph, and the operators inside a subgraph handle format conversion themselves if necessary. This makes interaction of TVM and MKLDNN with MXNet much easier. Neither the MXNet executor nor the MXNet operators need to deal with customized data formats. Even though invoking these libraries from MXNet requires similar steps, the partitioning rule and the subgraph execution of these backends can be different. As such, we define the following interface for backends to customize graph partitioning and subgraph execution inside an operator. More details can be found at PR 12157 and [Subgraph API](https://cwiki.apache.org/confluence/display/MXNET/Unified+integration+with+external+backend+libraries).

#### JVM Memory Management

The MXNet Scala and Java API uses native memory to manage NDArray, Symbol, Executor, DataIterators using MXNet's internal C APIs.  The C APIs provide appropriate interfaces to create, access and free these objects. MXNet Scala has corresponding Wrappers and APIs that have pointer references to the native memory. Before this project, JVM users (e.g. Scala, Clojure, or Java) of MXNet have to manage MXNet objects manually using the dispose pattern. There are a few usability problems with this approach:

* Users have to track the MXNet objects manually and remember to call `dispose`. This is not Java idiomatic and not user friendly. Quoting a user: "this feels like I am writing C++ code which I stopped ages ago".
* Leads to memory leaks if `dispose` is not called.
* Many objects in MXNet-Scala are managed in native memory, needing to use `dispose` on them as well.
* Bloated code with `dispose()` methods.
* Hard to debug memory-leaks.
Goals of the project are:
* Provide MXNet JVM users automated memory management that can release native memory when there are no references to JVM objects.
* Provide automated memory management for both GPU and CPU memory without performance degradation.  More details can be found here: [JVM Memory Management](https://cwiki.apache.org/confluence/display/MXNET/JVM+Memory+Management)

#### Topology-aware AllReduce (experimental)
For distributed training, the `Reduce` communication patterns used by NCCL and MXNet are not optimal for small batch sizes. The `Topology-aware AllReduce` approach is based on the idea of using trees to perform the `Reduce` and `Broadcast` operations. We can use the idea of minimum spanning trees to do a binary tree `Reduce` communication pattern to improve distributed training following this paper by Wang, Li, Edo and Smola [1]. Our strategy is to use:

  * a single tree (latency-optimal for small messages) to handle `Reduce` on small messages
  * multiple trees (bandwidth-optimal for large messages) to handle `Reduce` on large messages

More details can be found here: [Topology-aware AllReduce](https://cwiki.apache.org/confluence/display/MXNET/Single+machine+All+Reduce+Topology-aware+Communication)
Note: This is an experimental feature and has known problems - see [13341](https://github.com/apache/incubator-mxnet/issues/13341). Please help to contribute to improve the robustness of the feature.

#### MKLDNN backend: Graph optimization and Quantization (experimental)

Two advanced features, graph optimization (operator fusion) and reduced-precision (INT8) computation, are introduced to MKLDNN backend in this release ([#12530](https://github.com/apache/incubator-mxnet/pull/12530), [#13297](https://github.com/apache/incubator-mxnet/pull/13297), [#13260](https://github.com/apache/incubator-mxnet/pull/13260)).
These features significantly boost the inference performance on CPU (up to 4X) for a broad range of deep learning topologies. Currently, this feature is only available for inference on platforms with [supported Intel CPUs](https://github.com/intel/mkl-dnn#system-requirements).

##### Graph Optimization
MKLDNN backend takes advantage of MXNet subgraph to implement the most of possible operator fusions for inference, such as Convolution + ReLU, Batch Normalization folding, etc. When using mxnet-mkl package, users can easily enable this feature by setting export MXNET_SUBGRAPH_BACKEND=MKLDNN.

##### Quantization
Performance of reduced-precision (INT8) computation is also dramatically improved after the graph optimization feature is applied on CPU Platforms. Various models are supported and can benefit from reduced-precision computation, including symbolic models, Gluon models and even custom models. Users can run most of the pre-trained models with only a few lines of commands and a new quantization script imagenet_gen_qsym_mkldnn.py. The observed accuracy loss is less than 0.5% for popular CNN networks, like ResNet-50, Inception-BN, MobileNet, etc.

Please find detailed information and performance/accuracy numbers here: [MKLDNN README](https://github.com/apache/incubator-mxnet/blob/master/MKLDNN_README.md), [quantization README](https://github.com/apache/incubator-mxnet/tree/master/example/quantization#1) and [design proposal](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN)

### New Operators

* Add trigonometric operators (#12424)
* [MXNET-807] Support integer label type in ctc_loss operator (#12468)
* [MXNET-876] make CachedOp a normal operator (#11641)
* Add index_copy() operator (#12810)
* Fix getnnz operator for CSR matrix (#12908) - issue #12872
* [MXNET-1173] Debug operators - isfinite, isinf and isnan (#12967)
* Add sample_like operators (#13034)
* Add gauss err function operator (#13229)
* [MXNET -1030] Enhanced Cosine Embedding Loss (#12750)
* Add bytearray support back to imdecode (#12855, #12868) (#12912)
* Add Psroipooling CPU implementation (#12738)

### Feature improvements
#### Operator
* [MXNET-912] Refactoring ctc loss operator (#12637)
* Refactor L2_normalization (#13059)
* Customized and faster `TakeOpForward` operator on CPU (#12997)
* Allow stop of arange operator to be inferred from dims. (#12064)
* Make check_isfinite, check_scale optional in clip_global_norm (#12042) add FListInputNames attribute to softmax_cross_entropy (#12701) [MXNET-867] Pooling1D with same padding (#12594)
* Add support for more req patterns for bilinear sampler backward (#12386) [MXNET-882] Support for N-d arrays added to diag op. (#12430)

#### Optimizer
* Add a special version of Adagrad optimizer with row-wise learning rate (#12365)
* Add a Python SVRGModule for performing SVRG Optimization Logic (#12376)

#### Sparse

* Fall back when sparse arrays are passed to MKLDNN-enabled operators (#11664)
* Add Sparse support for logic operators (#12860)
* Add Sparse support for take(csr, axis=0)  (#12889)

#### ONNX

* ONNX export - Clip operator (#12457)
* ONNX version update from 1.2.1 to 1.3 in CI (#12633)
* Use modern ONNX API to load a model from file (#12777)
* [MXNET-892] ONNX export/import: DepthToSpace, SpaceToDepth operators (#12731)
* ONNX export: Fully connected operator w/o bias, ReduceSum, Square (#12646)
* ONNX export/import: Selu (#12785)
* ONNX export: Cleanup (#12878)
* [MXNET-892] ONNX export/import: DepthToSpace, SpaceToDepth operators (#12731)
* ONNX export: Scalar, Reshape - Set appropriate tensor type (#13067)
* [MXNET-886] ONNX export: HardSigmoid, Less, Greater, Equal (#12812)

#### MKLDNN

* MKLDNN Forward FullyConnected  op cache (#11611)
* [MXNET-753] Fallback when using non-MKLDNN supported operators (#12019)
* MKLDNN Backward op cache (#11301)
* Implement mkldnn convolution fusion and quantization. (#12530)
* Improve mkldnn fallback. (#12663)
* Update MKL-DNN dependency (#12953)
* Update MKLML dependency (#13181)
* [MXNET-33] Enhance mkldnn pooling to support full convention (#11047)

#### Inference
* [MXNET-910] Multithreading inference. (#12456)
* Tweaked the copy in c_predict_api.h (#12600)

#### Other
* support for upper triangular matrices in linalg (#12904)
* Introduce Random module / Refactor code generation (#13038)
* [MXNET-779]Add DLPack Transformation API (#12047)
* Draw label name next to corresponding bounding boxes when the mapping of id to names is specified (#9496)
* Track epoch metric separately (#12182)
* Set correct update on kvstore flag in dist_device_sync mode (#12786)

### Frontend API updates

#### Gluon

* Update basic_layers.py (#13299)
* Gluon LSTM Projection and Clipping Support (#13056)
* Make Gluon download function to be atomic (#12572)
* [MXNET -1004] Poisson NegativeLog Likelihood loss (#12697)
* Add activation information for `mxnet.gluon.nn._Conv` (#12354)
* Gluon DataLoader: avoid recursionlimit error (#12622)

#### Symbol
* Addressed dumplicate object reference issues (#13214)
* Throw exception if MXSymbolInferShape fails (#12733)
* Infer dtype in SymbolBlock import from input symbol (#12412)

### Language API updates
#### Java
* [MXNET-1198] MXNet Java API (#13162)

#### R
* Refactor R Optimizers to fix memory leak - 11374
* Add new Vignettes to the R package
  * Char-level Language modeling - 12670
  * Multidimensional Time series forecasting - 12664
* Fix broken Examples and tutorials
  * Tutorial on neural network introduction - 12117
  * CGAN example - 12283
  * Test classification with LSTMs - 12263

#### Scala
* Explain the details for Scala Experimental (#12348)
* [MXNET-716] Adding Scala Inference Benchmarks (#12721)
* [MXNET-716][MIRROR #12723] Scala Benchmark Extension pack (#12758)
* NativeResource Management in Scala (#12647)
* Ignore generated Scala files (#12928)
* Use ResourceScope in Model/Trainer/FeedForward.scala (#12882)
* [MXNET-1180] Scala Image API (#12995)
* Update log4j version of Scala package (#13131)
* Review require() usages to add meaningful messages (#12570)
* Fix Scala readme (#13082)

#### Clojure
* Introduction to Clojure-MXNet video link (#12754)
* Improve the Clojure Package README to Make it Easier to Get Started (#12881)
* MXNET-873 - Bring Clojure Package Inline with New DataDesc and Layout in Scala Package (#12387)
* Port of Scala Image API to Clojure (#13107)

#### Perl
* [MXNET-1026] [Perl] Sync with recent changes in Python's API (#12739)

#### Julia
* Import Julia binding (#10149), how to use is available at https://github.com/apache/incubator-mxnet/tree/master/julia

### Performance benchmarks and improvements
* Update mshadow for omp acceleration when nvcc is not present  (#12674)
* [MXNET-860] Avoid implicit double conversions (#12361)
* Add more models to benchmark_score (#12780)
* Add resnet50-v1 to benchmark_score (#12595)

### Bug fixes
* Fix for #10920 -  increase tolerance for sparse dot (#12527)
* [MXNET-1234] Fix shape inference problems in Activation backward (#13409)
* Fix a bug in `where` op with 1-D input (#12325)
* [MXNET-825] Fix CGAN R Example with MNIST dataset (#12283)
* [MXNET-535] Fix bugs in LR Schedulers and add warmup (#11234)
* Fix speech recognition example (#12291)
* Fix bug in 'device' type kvstore (#12350)
* fix search result 404s (#12414)
* Fix help in imread (#12420)
* Fix render issue on &lt; and &gt; (#12482)
* [MXNET-853] Fix for smooth_l1 operator scalar default value (#12284)
* Fix subscribe links, remove disabled icons (#12474)
* Fix broken URLs (#12508)
* Fix/public internal header (#12374)
* Fix lazy record io when used with dataloader and multi_worker > 0 (#12554)
* Fix error in try/finally block for blc (#12561)
* Add cudnn_off parameter to SpatialTransformer Op and fix the inconsistency between CPU & GPU code (#12557)
* [MXNET-798] Fix the dtype cast from non float32 in Gradient computation (#12290)
* Fix CodeCovs proper commit detection (#12551)
* Add TensorRT tutorial to index and fix ToC (#12587)
* Fixed typo in c_predict_api.cc (#12601)
* Fix typo in profiler.h (#12599)
* Fixed NoSuchMethodError for Jenkins Job for MBCC (#12618)
* [MXNET-922] Fix memleak in profiler (#12499)
* [MXNET-969] Fix buffer overflow in RNNOp (#12603)
*  Fixed param coercion of clojure executor/forward (#12627) (#12630)
* Fix version dropdown behavior (#12632)
* Fix reference to wrong function (#12644)
* Fix the location of the tutorial of control flow operators (#12638)
* Fix issue 12613 (#12614)
* [MXNET-780] Fix exception handling bug (#12051)
* Fix bug in prelu, issue 12061 (#12660)
* [MXNET-833] [R] Char-level RNN tutorial fix (#12670)
* Fix static / dynamic linking of gperftools and jemalloc (#12714)
* Fix #12672, importing numpy scalars (zero-dimensional arrays) (#12678)
* [MXNET-623] Fixing an integer overflow bug in large NDArray (#11742)
* Fix benchmark on control flow operators (#12693)
* Fix regression in MKLDNN caused by PR 12019 (#12740)
* Fixed broken link for Baidu's WARP CTC (#12774)
* Fix CNN visualization tutorial (#12719)
* [MXNET-979] Add fix_beta support in BatchNorm (#12625)
* R fix metric shape (#12776)
* Revert [MXNET-979] Add fix_beta support in BatchNorm (#12625) (#12789)
* Fix mismatch shapes (#12793)
* Fixed symbols naming in RNNCell, LSTMCell, GRUCell (#12794)
* Fixed __setattr__ method of _MXClassPropertyMetaClass (#12811)
* Fixed regex for matching platform type in Scala Benchmark scripts (#12826)
* Fix broken links (#12856)
* Fix Flaky Topk (#12798)
* [MXNET-1033] Fix a bug in MultiboxTarget GPU implementation (#12840)
* [MXNET-1107] Fix CPUPinned unexpected behaviour (#12031)
* Fix __all__ in optimizer/optimizer.py (#12886)
* Fix Batch input issue with Scala Benchmark (#12848)
* fix type inference in index_copy. (#12890)
* Fix the paths issue for downloading script (#12913)
* Fix indpt[0] for take(csr) (#12927)
* Fix the bug of assigning large integer to NDArray (#12921)
* Fix Sphinx errors for tutorials and install ToCs (#12945)
* Fix variable name in tutorial code snippet (#13052)
* Fix example for mxnet.nd.contrib.cond and fix typo in src/engine (#12954)
* Fix a typo in operator guide (#13115)
* Fix variational autoencoder example (#12880)
* Fix problem with some OSX not handling the cast on imDecode (#13207)
* [MXNET-953] Fix oob memory read (#12631)
* Fix Sphinx error in ONNX file (#13251)
* [Example] Fixing Gradcam implementation (#13196)
* Fix train mnist for inception-bn and resnet (#13239)
* Fix a bug in index_copy (#13218)
* Fix Sphinx errors in box_nms (#13261)
* Fix Sphinx errors (#13252)
* Fix the cpp example compiler flag (#13293)
* Made fixes to sparse.py and sparse.md (#13305)
* [Example] Gradcam- Fixing a link (#13307)
* Manually track num_max_thread (#12380)
* [Issue #11912] throw mxnet exceptions when decoding invalid images. (#12999)
* Undefined name: load_model() --> utils.load_model() (#12867)
* Change the way NDArrayIter handle the last batch (#12545)
* Add embedding to print_summary (#12796)
* Allow foreach on input with 0 length (#12471)
* [MXNET-360]auto convert str to bytes in img.imdecode when py3 (#10697)
* Fix unpicklable transform_first on windows (#13686)

### Licensing updates
* Add license headers to R-package (#12559)
* License header (#13178)
* add url and license to clojure package project (#13304)

### Improvements
#### Tutorial
* [MXNET-422] Distributed training tutorial (#10955)
* Add a tutorial for control flow operators. (#12340)
* Add tutorial Gotchas using NumPy (#12007)
* Updated Symbol tutorial with Gluon (#12190)
* Improve tutorial redirection (#12607)
* Include missing import in TensorRT tutorial (#12609)
* Update Operator Implementation Tutorial (#12230)
* Add a tutorial for the subgraph API. (#12698)
* Improve clojure tutorial (#12974)
* Update scala intellij tutorial (#12827)
* [Example] Gradcam consolidation in tutorial (#13255)
* [MXNET-1203] Tutorial infogan  (#13144)
* [MXNET-703] Add a TensorRT walkthrough (#12548)

#### Example
* Update C++ example so it is easier to run (#12397)
* [MXNET-580] Add SN-GAN example (#12419)
* [MXNET-637] Multidimensional LSTM example for MXNetR (#12664)
* [MXNET-982] Provide example to illustrate usage of CSVIter in C++ API (#12636)
* [MXNET-947] Expand scala imclassification example with resnet (#12639)
* MKL-DNN Quantization Examples and README (#12808)
* Extending the DCGAN example implemented by gluon API to provide a more straight-forward evaluation on the generated image (#12790)
* [MXNET-1017] Updating the readme file for cpp-package and adding readme file for example directory. (#12773)
* Update tree lstm example (#12960)
* Update bilstm integer array sorting example (#12929)
* Updated / Deleted some examples (#12968)
* Update module example (#12961)
* Update adversary attack generation example (#12918)
* Update Gluon example folder (#12951)
* Update dec example (#12950)
* Updated capsnet example (#12934)
* Updates to several examples (#13068)
* Update multi-task learning example (#12964)
* Remove obsolete memory cost example (#13235)
* [Example] Update cpp example README (#13280)
* [Example]update NER example readme on module prediction (#13184)
* Update proposal_target.py (#12709)
* Removing the re-size for validation data, which breaking the validation accuracy of CIFAR training (#12362)
* Update the README with instruction to redirect the user to gluon-cv (#13186)

#### Documentation
* Update ONNX API docs references (#12317)
* Documentation update related to sparse support (#12367)
* Edit shape.array doc and some style improvements (#12162)
* Fixed docs/website build checkout bug (#12413)
* Add Python API docs for test_utils and visualization (#12455)
* Fix the installation doc for MKL-DNN backend (#12534)
* Added comment to docs regarding ToTensor transform (#12186)
* Pinned dockcross to a tag with fixed ABI for RPi (#12588)
* Refine the documentation of im2rec (#12606)
* Update and modify Windows docs (#12620)
* update docs to list cmake required for build from source page (#12592)
* update the distributed_training document (#12626)
* Add docstring in im2rec.py (#12621)
* [Doc] Change the description for pip packages (#12584)
* Change dependencies documentation opencv2-->opencv (#12654)
* Add documents for two new environment variables for memory pool. (#12668)
* Scala Docs - Replace old Symbol api usages (#12759)
* add/update infer_range docs (#12879)
* Fix typo in formula in docstring for GRU cell and layer and add clarification to description (gluon.rnn) (#12896)
* Fix the operator API documentation (#12942)
* fix broken docs (#12871)
* fix mac r install and windows python build from source docs (#12919)
* Document the newly added env variable (#13049)
* Add documentation on GPU performance on Quantization example (#13145)
* Fix Sphinx python docstring formatting error. (#13177)
* [Doc] Fix repo paths in Ubuntu build doc (#13101)
* Fix Sphinx document parsing error. (#13195)
* Fix #13090, Add image.imread to python API doc. (#13176)
* Fix Sphinx docstring formatting error. (#13004, #13005, #13006) (#13175)
* Fix #12944, Fix Sphinx python docstring formatting error. (#13174)
* Fix #13013, Fix Sphinx python docstring error. (#13173)
* Fixed Sparse astype doc string formatting error (#13171)
* Fixed Documentation issues (#13215)
* update the doc (#13205)
* Fix Sphinx doc errors (#13170)
* Fix Sphinx python docstring error: initializer.InitDesc (#12939) (#13148)
* Fix Sphinx python docstring error: text contrib module (#12949) (#13149)
* Fix Sphinx python docstrings (#13160)
* Add Java API docs generation (#13071)
* Fix scaladoc build errors (#13189)
* Add missing documentations for getnnz (#13128)
* Addressed ONNX module documentation warnings and added notes for short-form representation (#13259)
* Doc fixes (#13256)
* Addressed doc issues (#13165)
* stop gap fix to let website builds through; scaladoc fix pending (#13298)
* Fix Sphinx python docstring formatting error. (#13194)
* Visualization doc fix. Added notes for shortform (#13291)
* [Example] Add docstring for test optimizer and test score (#13286)
* Fix descriptions in scaladocs for macro ndarray/sybmol APIs (#13210)
* Sphinx error reduction (#12323)
* Sphinx errors in Gluon (#13275)
* Update env_var.md (#12702)
* Updated the Instructions for use of the label bot (#13192)
* Added/changed file_name, brief description comments in some files (#13033)

#### Website
* adding apache conf promo to home page (#12347)
* Consistent website theme and custom 404 (#12426)
* update apachecon links to https (#12521)
* [HOLD] 1.3.0 release website updates (#12509)
* add mentions of the gluon toolkits and links to resources (#12667)
* remove apachecon promo (#12695)
* [MXNet-1002] Add GluonCV and NLP tookits, Keras, and developer wiki to navigation (#12704)

#### MXNet Distributions
* Make the output of ci/docker/install/ubuntu_mklml.sh less verbose (#12422)
* Fix tvm dependency for docker (#12479)
* [MXNET-703] Add TensorRT runtime Dockerfile (#12549)
* [MXNET-951] Python dockerfiles built on pip binaries and build/release script (#12556)
* Change numpy version to 1.15.2 in python and docker install requirements (#12711)
* Add mkl-dnn to docker install method (#12643)
* Fix docker cleanup race condition (#13092)
* Bugfix in ci/docker_cache.py (#13249)
* Update PyPI version number (#11773)
* update download links to apache distros (#12617)

#### Installation
* Installation instructions consolidation (#12388)
* Refine mxnet python installation (#12696)
* R install instructions update for macOS (#12832)
* remove legacy installation of Roxygen2 5.0 and add R-specific clean target (#12993) (#12998)
* Force APT cache update before executing install (#13285)
* Make the Ubuntu scripts executable after download. (#12180)
* replacing windows setup with newer instructions (#12504)
* Updated download links and verification instructions (#12651)
* Remove pip overwrites (#12604)

#### Build and CI
* [MXNET-908] Enable minimal OSX Travis build (#12462)
* Use jom for parallel Windows builds (#12533)
* [MXNET-950] Enable parallel R dep builds in CI (#12552)
* Speed up CI Windows builds (#12563)
* [MXNET-908] Speed up travis builds to avoid timeouts (#12706)
* Simplify mac MKLDNN build (#12724)
* [MXNET-674] Speed up GPU builds in CI (#12782)
* Improved git reset for CI builds (#12784)
* Improve cpp-package example project build files. (#13093)
* Add --no-cache option to build.py when building containers (#13182)
* Addressed sphinx build issue (#13246)
* Tighten up PyLint directives again (#12322)
* [MXNET-859] Add a clang-tidy stage to CI (#12282)
* A solution to prevent zombie containers locally and in CI (#12381)
*  [MXNET-696][PYTHON][UNDEFINED NAME] import logging in ci/util.py (#12488)
* [MXNET-703] Static linking for libprotobuf with TensorRT (#12475)
* Remove regression checks for website links (#12507)
* [MXNET-953] - Add ASAN sanitizer, Enable in CI (#12370)
* Allow custom path and static linking for custom mallocs in make (#12645)
* Correct PR branch detection in code coverage (#12615)
* Update osx.mk - Added apple to USE_BLAS comment (#12819)
* [MXNET-953] Correct ASAN cflags flag (#12659)
* [MXNET-1025] Add Jetpack 3.3 support to Jetson (#12735)
* Fail the broken link job when broken links are found (#12905)
* Removed unused header (#13066)
* Maven Surefire bug workaround (#13081)
* Add Turing and Volta support to arch_name (#13168)
* Moves f16c autodetection to its own cmake module (#12331)
* la_op_inline.h to la_op-inl.h for consistency (#13045)
* [MXNET-793] Virtualized ARMv7 with Qemu CI integration (#13203)
* Remove unused variable `rotateM_` (#10803)
* Separate refactoring from #12276 in a prior PR (#12296)
* [MXNET-860] Remove std::moves that have no affect (#12730)
* [MXNET-860] Use emplace where helpful (#12694)
* Enable C++ coverage (#12642)
* [MXNET-860] Update to modern nullptr usage (#12352)
* [MXNET-860] Reduce redundant copies, check for regressions with clang-tidy (#12355)


#### 3rd party
##### TVM:
* Updated tvm submodule head (#12764)
* Updated tvm submodule head (#12448)
##### CUDNN:
* [MXNET-1179] Enforce deterministic algorithms in convolution layers (#12992)
* CudnnFind() usage improvements (#12804)
* Add option for automatic downcasting dtype for cudnn to allow using Tensorcore for fp32  (#12722)
##### Horovod:
* [MXNET-1111] Remove CPUPinned in ImageRecordIter (#12666)

### Deprications
* Add a deprecate message (#13042) contrib_CTCLoss is deprecated. Added a message in command
### Other
* Updating news, readme files and bumping master version to 1.3.1 (#12525)
* Add new name to CONTRIBUTORS.md (#12763)
* Update contribute.md (#12685)
* Updated CONTRIBUTORS.md to include lebeg and gigasquid, moved mabreu to committers section (#12766)
* Update CONTRIBUTORS.md (#12996)
* Updated CONTRIBUTORS.md to include mxnet-label-bot  (#13048)

### How to build MXNet
Please follow the instructions at https://mxnet.incubator.apache.org/install/index.html

### List of submodules used by Apache MXNet (Incubating) and when they were updated last
Submodule@commit ID::Last updated by MXNet:: Last update in submodule

* cub@05eb57f::Jul 31, 2017 :: Jul 31, 2017
* dlpack@10892ac:: Oct 30, 2017 :: Aug 23, 2018
* dmlc-core@0a0e8ad:: Aug 15, 2018 :: Nov 15, 2018
* googletest@ec44c6c:: July 14, 2016 :: July 14, 2016
* mkldnn@a7c5f53:: Nov 7, 2018 :: Nov 5, 2018
* mshadow@696803b:: Sep 28, 2018 :: Nov 7,  2018
* onnx-tensorrt@3d8ee04:: Aug 22, 2018 :: Nov 10, 2018
* openmp@37c7212: Nov 22, 2017 :: Nov 13, 2018
* ps-lite@8a76389: April 25, 2018 :: Oct 9, 2018
* tvm@0f053c8: Oct 10, 2018 :: Oct 8, 2018

## 1.3.1

### Bug fixes

* [MXNET-953] Fix oob memory read (v1.3.x) / [#13118](https://github.com/apache/incubator-mxnet/pull/13118)  
Simple bugfix addressing an out-of-bounds memory read.


* [MXNET-969] Fix buffer overflow in RNNOp (v1.3.x) / [#13119](https://github.com/apache/incubator-mxnet/pull/13119)  
This fixes an buffer overflow detected by ASAN.


* CudnnFind() usage improvements (v1.3.x) / [#13123](https://github.com/apache/incubator-mxnet/pull/13123)  
  This PR improves the MXNet's use of cudnnFind() to address a few issues:
  1. With the gluon imperative style, cudnnFind() is called during forward(), and so might have its timings perturbed by other GPU activity (including potentially other cudnnFind() calls).
  2. With some cuda drivers versions, care is needed to ensure that the large I/O and workspace cudaMallocs() performed by cudnnFind() are immediately released and available to MXNet.
  3. cudnnFind() makes both conv I/O and workspace allocations that must be covered by the GPU global memory headroom defined by MXNET_GPU_MEM_POOL_RESERVE. Per issue #12662, large convolutions can result in out-of-memory errors, even when MXNet's storage allocator has free memory in its pool.  
  
  This PR addresses these issues, providing the following benefits:
  1. Consistent algo choice for a given convolution type in a model, both for instances in the same GPU and in other GPUs in a multi-GPU training setting.
  2. Consistent algo choice from run to run, based on eliminating sources of interference of the cudnnFind() timing process.
  3. Consistent model global memory footprint, both because of the consistent algo choice (algo's can have markedly different workspace requirements) and changes to MXNet's use of cudaMalloc.
  4. Increased training performance based on being able to consistently run with models that approach the GPU's full global memory footprint.
  5. Adds a unittest for and solves issue #12662.

* [MXNET-922] Fix memleak in profiler (v1.3.x) / [#13120](https://github.com/apache/incubator-mxnet/pull/13120)  
  Fix a memleak reported locally by ASAN during a normal inference test.

* Fix lazy record io when used with dataloader and multi_worker > 0 (v1.3.x) / [#13124](https://github.com/apache/incubator-mxnet/pull/13124)  
  Fixes multi_worker data loader when record file is used. The MXRecordIO instance needs to require a new file handler after fork to be safely manipulated simultaneously.

  This fix also safely voids the previous temporary fixes #12093 #11370.

* fixed symbols naming in RNNCell, LSTMCell, GRUCell (v1.3.x) / [#13158](https://github.com/apache/incubator-mxnet/pull/13158)  
  This fixes #12783, by assigning all nodes in hybrid_forward a unique name. Some operations were in fact performed without attaching the appropriate (time) prefix to the name, which makes serialized graphs non-deserializable.

* Fixed `__setattr__` method of `_MXClassPropertyMetaClass` (v1.3.x) / [#13157](https://github.com/apache/incubator-mxnet/pull/13157)  
  Fixed `__setattr__` method

* allow foreach on input with 0 length (v1.3.x) / [#13151](https://github.com/apache/incubator-mxnet/pull/13151)  
  Fix #12470. With this change, outs shape can be inferred correctly.

* Infer dtype in SymbolBlock import from input symbol (v1.3.x) / [#13117](https://github.com/apache/incubator-mxnet/pull/13117)  
  Fix for the issue - #11849  
  Currently, Gluon symbol block cannot import any symbol with type other than fp32. All the parameters are created as FP32 leading to failure in importing the params when it is of type fp16, fp64 etc,  
  In this PR, we infer the type of the symbol being imported and create the Symbol Block Parameters with that inferred type.  
  Added the tests

### Documentation fixes

* Document the newly added env variable (v1.3.x) / [#13156](https://github.com/apache/incubator-mxnet/pull/13156)  
  Document the env variable: MXNET_ENFORCE_DETERMINISM added in PR: [#12992](https://github.com/apache/incubator-mxnet/pull/12992)

* fix broken links (v1.3.x) / [#13155](https://github.com/apache/incubator-mxnet/pull/13155)  
  This PR fixes broken links on the website.

* fix broken Python IO API docs (v1.3.x) / [#13154](https://github.com/apache/incubator-mxnet/pull/13154)  
  Fixes [#12854: Data Iterators documentation is broken](https://github.com/apache/incubator-mxnet/issues/12854)

  This PR manually specifies members of the IO module so that the docs will render as expected. This is workaround in the docs to deal with a bug introduced in the Python code/structure since v1.3.0. See the comments for more info.

  This PR also fixes another issue that may or may not be related. Cross references to same-named entities like name, shape, or type are confusing Sphinx and it seems to just link to whatever it last dealt with that has the same name, and not the current module. To fix this you have to be very specific. Don't use type, use np.type if that's what you want. Otherwise you might end up with mxnet.kvstore.KVStore.type. This is a known Sphinx issue, so it might be something we have to deal with for the time being.

  This is important for any future modules - that they recognize this issue and make efforts to map the params and other elements.

* add/update infer_range docs (v1.3.x) / [#13153](https://github.com/apache/incubator-mxnet/pull/13153)  
  This PR adds or updates the docs for the infer_range feature.

  Clarifies the param in the C op docs
  Clarifies the param in the the Scala symbol docs
  Adds the param for the the Scala ndarray docs
  Adds the param for the Python symbol docs
  Adds the param for the Python ndarray docs

### Other Improvements

* [MXNET-1179] Enforce deterministic algorithms in convolution layers (v1.3.x) / [#13152](https://github.com/apache/incubator-mxnet/pull/13152)  
  Some of the CUDNN convolution algorithms are non-deterministic (see issue #11341). This PR adds an env variable to enforce determinism in the convolution operators. If set to true, only deterministic CUDNN algorithms will be used. If no deterministic algorithm is available, MXNet will error out.


### Submodule updates

* update mshadow (v1.3.x) / [#13122](https://github.com/apache/incubator-mxnet/pull/13122)  
  Update mshadow for omp acceleration when nvcc is not present

### Known issues

The test test_operator.test_dropout has issues and has been disabled on the branch:

* Disable flaky test test_operator.test_dropout (v1.3.x) / [#13200](https://github.com/apache/incubator-mxnet/pull/13200)



For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/x/eZGzBQ)


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
