<!--
  ~ Licensed to the Apache Software Foundation (ASF) under one
  ~ or more contributor license agreements.  See the NOTICE file
  ~ distributed with this work for additional information
  ~ regarding copyright ownership.  The ASF licenses this file
  ~ to you under the Apache License, Version 2.0 (the
  ~ "License"); you may not use this file except in compliance
  ~ with the License.  You may obtain a copy of the License at
  ~
  ~   http://www.apache.org/licenses/LICENSE-2.0
  ~
  ~ Unless required by applicable law or agreed to in writing,
  ~ software distributed under the License is distributed on an
  ~ "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
  ~ KIND, either express or implied.  See the License for the
  ~ specific language governing permissions and limitations
  ~ under the License.
  ~
-->

MXNet Change Log
================
- [MXNet Change Log](#mxnet-change-log)
  - [1.8.0](#180)
    - [Features](#features)
      * [CUDA Graphs](#cuda-graphs)
      * [CUDA 11 Support](#cuda-11-support)
      * [TensorRT](#tensorrt)
      * [OneDNN](#onednn)
      * [IntGemm](#intgemm)
      * [Subgraph API](#subgraph-api)
      * [Extensions](#extensions)
      * [ONNX](#onnx)
      * [Large Tensor](#large-tensor)
      * [Website Improvements](#website-improvements)
      * [Documentation](#documentation)
      * [License](#license)
      * [CI Improvements](#ci-improvements)
      * [Bug Fixes](#bug-fixes)
  - [1.7.0](#170)
    - [New features](#new-features)
      - [MXNet Extensions: custom operators, partitioning, and graph passes](#mxnet-extensions-custom-operators-partitioning-and-graph-passes)
      - [OpPerf utility enabled in the binary distribution](#opperf-utility-enabled-in-the-binary-distribution)
      - [MKL-DNN](#mkl-dnn)
        - [MKL-DNN as the default CPU backend in binary distribution](#mkl-dnn-as-the-default-cpu-backend-in-binary-distribution)
        - [Branding change to DNNL](#branding-change-to-dnnl)
        - [Support bfloat16 datatype](#support-bfloat16-datatype)
      - [New operators](#new-operators)
    - [Feature improvements](#feature-improvements)
      - [Numpy compatible interface(experimental)](#numpy-compatible-interfaceexperimental)
      - [Large tensor support](#large-tensor-support)
      - [MKL-DNN enhancement](#mkl-dnn-enhancement)
      - [TensorRT integration](#tensorrt-integration)
      - [Quantization](#quantization)
      - [Profiler](#profiler)
      - [ONNX](#onnx)
      - [New models](#new-models)
      - [Operator improvements](#operator-improvements)
      - [Bug fixes](#bug-fixes)
    - [Front end API](#front-end-api)
      - [Gluon](#gluon)
      - [Symbol](#symbol)
    - [Language Bindings](#language-bindings)
      - [Python](#python)
      - [C/C++](#cc)
      - [R](#r)
      - [Clojure](#clojure)
      - [Julia](#julia)
      - [Perl](#perl)
      - [Scala](#scala)
    - [Performance improvements](#performance-improvements)
    - [Example and tutorials](#example-and-tutorials)
    - [Website and documentation](#website-and-documentation)
    - [CI/CD](#cicd)
    - [License](#license)
    - [Miscellaneous changes](#miscellaneous-changes)
  - [1.6.0](#160)
    - [Deprecation of Python 2](#deprecation-of-python-2)
    - [New features](#new-features-1)
      - [NumPy compatible interface and using TVM to generate operators](#numpy-compatible-interface-and-using-tvm-to-generate-operators)
      - [Graph optimizations](#graph-optimizations)
        - [Pointwise fusion for GPU](#pointwise-fusion-for-gpu)
        - [Eliminate common subexpressions](#eliminate-common-subexpressions)
        - [Default MKLDNN Subgraph fusion](#default-mkldnn-subgraph-fusion)
      - [New operators](#new-operators-1)
    - [Feature improvements](#feature-improvements-1)
      - [Automatic Mixed Precision](#automatic-mixed-precision)
      - [Gluon Fit API](#gluon-fit-api)
      - [MKLDNN](#mkldnn)
      - [Large tensor support](#large-tensor-support-1)
      - [TensorRT integration](#tensorrt-integration-1)
      - [Higher order gradient support](#higher-order-gradient-support)
      - [Operator improvements](#operator-improvements-1)
      - [Profiler](#profiler-1)
      - [ONNX import/export](#onnx-importexport)
      - [Runtime discovery of features](#runtime-discovery-of-features)
      - [Bug fixes](#bug-fixes-1)
    - [Front end API](#front-end-api-1)
      - [Gluon](#gluon-1)
      - [Symbol](#symbol-1)
    - [Language Bindings](#language-bindings-1)
      - [Python](#python-1)
      - [C/C++](#cc-1)
      - [Clojure](#clojure-1)
      - [Julia](#julia-1)
      - [Perl](#perl-1)
      - [Scala](#scala-1)
    - [Performance improvements](#performance-improvements-1)
    - [Examples and tutorials](#examples-and-tutorials)
    - [Website and documentation](#website-and-documentation-1)
    - [CI/CD](#cicd-1)
    - [Misc](#misc)
  - [1.5.1](#151)
    - [Bug-fixes](#bug-fixes-2)
  - [1.5.0](#150)
    - [New Features](#new-features-2)
      - [Automatic Mixed Precision(experimental)](#automatic-mixed-precisionexperimental)
      - [MKL-DNN Reduced precision inference and RNN API support](#mkl-dnn-reduced-precision-inference-and-rnn-api-support)
      - [Dynamic Shape(experimental)](#dynamic-shapeexperimental)
      - [Large Tensor Support](#large-tensor-support-2)
      - [Dependency Update](#dependency-update)
      - [Gluon Fit API(experimental)](#gluon-fit-apiexperimental)
      - [New Operators](#new-operators-2)
    - [Feature Improvements](#feature-improvements-2)
      - [Operators](#operators)
      - [MKLDNN](#mkldnn-1)
      - [ONNX](#onnx-1)
      - [TensorRT](#tensorrt)
      - [FP16 Support](#fp16-support)
      - [Deep Graph Library(DGL) support](#deep-graph-librarydgl-support)
      - [Horovod Integration](#horovod-integration)
      - [Dynamic Shape](#dynamic-shape)
      - [Backend Engine](#backend-engine)
      - [Large Tensor Support](#large-tensor-support-3)
      - [Quantization](#quantization-1)
      - [Profiler](#profiler-2)
      - [CoreML](#coreml)
    - [Front End API](#front-end-api-2)
      - [Gluon](#gluon-2)
      - [Python](#python-2)
    - [Language Bindings](#language-bindings-2)
      - [Scala](#scala-2)
      - [Java](#java)
      - [C++](#c)
      - [Clojure](#clojure-2)
      - [Julia](#julia-2)
      - [Perl:](#perl-2)
      - [R](#r-1)
    - [Performance Improvements](#performance-improvements-2)
    - [Example and Tutorials](#example-and-tutorials-1)
    - [Website](#website)
    - [Documentation](#documentation)
    - [Build and Test](#build-and-test)
    - [Bug-fixes](#bug-fixes-3)
    - [License](#license-1)
    - [Depreciations](#depreciations)
    - [Known Issues](#known-issues)
  - [1.4.1](#141)
    - [Bug-fixes](#bug-fixes-4)
  - [1.4.0](#140)
    - [New Features](#new-features-3)
      - [Java Inference API](#java-inference-api)
      - [Julia API](#julia-api)
      - [Control Flow Operators (experimental)](#control-flow-operators-experimental)
      - [SVRG Optimization](#svrg-optimization)
      - [Subgraph API (experimental)](#subgraph-api-experimental)
      - [JVM Memory Management](#jvm-memory-management)
      - [Topology-aware AllReduce (experimental)](#topology-aware-allreduce-experimental)
      - [MKLDNN backend: Graph optimization and Quantization (experimental)](#mkldnn-backend-graph-optimization-and-quantization-experimental)
        - [Graph Optimization](#graph-optimization)
        - [Quantization](#quantization-2)
    - [New Operators](#new-operators-3)
    - [Feature improvements](#feature-improvements-3)
      - [Operator](#operator)
      - [Optimizer](#optimizer)
      - [Sparse](#sparse)
      - [ONNX](#onnx-2)
      - [MKLDNN](#mkldnn-2)
      - [Inference](#inference)
      - [Other](#other)
    - [Frontend API updates](#frontend-api-updates)
      - [Gluon](#gluon-3)
      - [Symbol](#symbol-2)
    - [Language API updates](#language-api-updates)
      - [Java](#java-1)
      - [R](#r-2)
      - [Scala](#scala-3)
      - [Clojure](#clojure-3)
      - [Perl](#perl-3)
      - [Julia](#julia-3)
    - [Performance benchmarks and improvements](#performance-benchmarks-and-improvements)
    - [Bug fixes](#bug-fixes-5)
    - [Licensing updates](#licensing-updates)
    - [Improvements](#improvements)
      - [Tutorial](#tutorial)
      - [Example](#example)
      - [Documentation](#documentation-1)
      - [Website](#website-1)
      - [MXNet Distributions](#mxnet-distributions)
      - [Installation](#installation)
      - [Build and CI](#build-and-ci)
      - [3rd party](#3rd-party)
        - [TVM:](#tvm)
        - [CUDNN:](#cudnn)
        - [Horovod:](#horovod)
    - [Deprications](#deprications)
    - [Other](#other-1)
    - [How to build MXNet](#how-to-build-mxnet)
    - [List of submodules used by Apache MXNet (Incubating) and when they were updated last](#list-of-submodules-used-by-apache-mxnet-incubating-and-when-they-were-updated-last)
  - [1.3.1](#131)
    - [Bug fixes](#bug-fixes-6)
    - [Documentation fixes](#documentation-fixes)
    - [Other Improvements](#other-improvements)
    - [Submodule updates](#submodule-updates)
    - [Known issues](#known-issues-1)
  - [1.3.0](#130)
    - [New Features - Gluon RNN layers are now HybridBlocks](#new-features---gluon-rnn-layers-are-now-hybridblocks)
    - [MKL-DNN improvements](#mkl-dnn-improvements)
    - [New Features - Gluon Model Zoo Pre-trained Models](#new-features---gluon-model-zoo-pre-trained-models)
    - [New Features - Clojure package (experimental)](#new-features---clojure-package-experimental)
    - [New Features - Synchronized Cross-GPU Batch Norm (experimental)](#new-features---synchronized-cross-gpu-batch-norm-experimental)
    - [New Features - Sparse Tensor Support for Gluon (experimental)](#new-features---sparse-tensor-support-for-gluon-experimental)
    - [New Features - Control flow operators (experimental)](#new-features---control-flow-operators-experimental)
    - [New Features - Scala API Improvements (experimental)](#new-features---scala-api-improvements-experimental)
    - [New Features - Rounding GPU Memory Pool for dynamic networks with variable-length inputs and outputs (experimental)](#new-features---rounding-gpu-memory-pool-for-dynamic-networks-with-variable-length-inputs-and-outputs-experimental)
    - [New Features - Topology-aware AllReduce (experimental)](#new-features---topology-aware-allreduce-experimental)
    - [New Features - Export MXNet models to ONNX format (experimental)](#new-features---export-mxnet-models-to-onnx-format-experimental)
    - [New Features - TensorRT Runtime Integration (experimental)](#new-features---tensorrt-runtime-integration-experimental)
    - [New Examples - Scala](#new-examples---scala)
    - [Maintenance - Flaky Tests improvement effort](#maintenance---flaky-tests-improvement-effort)
    - [Maintenance - MXNet Model Backwards Compatibility Checker](#maintenance---mxnet-model-backwards-compatibility-checker)
    - [Maintenance - Integrated testing for "the Straight Dope"](#maintenance---integrated-testing-for-%22the-straight-dope%22)
    - [Bug-fixes](#bug-fixes-7)
    - [Performance Improvements](#performance-improvements-3)
    - [API Changes](#api-changes)
    - [Other features](#other-features)
    - [Usability Improvements](#usability-improvements)
  - [1.2.0](#120)
    - [New Features - Added Scala Inference APIs](#new-features---added-scala-inference-apis)
    - [New Features - Added a Module to Import ONNX models into MXNet](#new-features---added-a-module-to-import-onnx-models-into-mxnet)
    - [New Features - Added Support for Model Quantization with Calibration](#new-features---added-support-for-model-quantization-with-calibration)
    - [New Features - MKL-DNN Integration](#new-features---mkl-dnn-integration)
    - [New Features - Added Exception Handling Support for Operators](#new-features---added-exception-handling-support-for-operators)
    - [New Features - Enhanced FP16 support](#new-features---enhanced-fp16-support)
    - [New Features - Added Profiling Enhancements](#new-features---added-profiling-enhancements)
    - [Breaking Changes](#breaking-changes)
    - [Bug Fixes](#bug-fixes-8)
    - [Performance Improvements](#performance-improvements-4)
    - [API Changes](#api-changes-1)
    - [Sparse Support](#sparse-support)
    - [Deprecations](#deprecations)
    - [Other Features](#other-features-1)
    - [Usability Improvements](#usability-improvements-1)
    - [Known Issues](#known-issues-2)
  - [1.1.0](#110)
    - [Usability Improvements](#usability-improvements-2)
    - [Bug-fixes](#bug-fixes-9)
    - [New Features](#new-features-4)
    - [API Changes](#api-changes-2)
    - [Deprecations](#deprecations-1)
    - [Performance Improvements](#performance-improvements-5)
    - [Known Issues](#known-issues-3)
  - [1.0.0](#100)
    - [Performance](#performance)
    - [New Features - Gradient Compression [Experimental]](#new-features---gradient-compression-experimental)
    - [New Features - Support of NVIDIA Collective Communication Library (NCCL) [Experimental]](#new-features---support-of-nvidia-collective-communication-library-nccl-experimental)
    - [New Features - Advanced Indexing [General Availability]](#new-features---advanced-indexing-general-availability)
    - [New Features - Gluon [General Availability]](#new-features---gluon-general-availability)
    - [New Features - ARM / Raspberry Pi support [Experimental]](#new-features---arm--raspberry-pi-support-experimental)
    - [New Features - NVIDIA Jetson support [Experimental]](#new-features---nvidia-jetson-support-experimental)
    - [New Features - Sparse Tensor Support [General Availability]](#new-features---sparse-tensor-support-general-availability)
    - [Bug-fixes](#bug-fixes-10)
    - [Doc Updates](#doc-updates)
  - [0.12.1](#0121)
    - [Bug-fixes](#bug-fixes-11)
  - [0.12.0](#0120)
    - [Performance](#performance-1)
    - [New Features - Gluon](#new-features---gluon)
    - [New Features - Autograd](#new-features---autograd)
    - [New Features - Sparse Tensor Support](#new-features---sparse-tensor-support)
    - [Other New Features](#other-new-features)
    - [API Changes](#api-changes-3)
    - [Bug-fixes](#bug-fixes-12)
  - [0.11.0](#0110)
    - [Major Features](#major-features)
    - [API Changes](#api-changes-4)
    - [Performance Improvements](#performance-improvements-6)
    - [Bugfixes](#bugfixes)
    - [Refactors](#refactors)
  - [0.10.0](#0100)
  - [0.9.3](#093)
  - [v0.8](#v08)
  - [v0.7](#v07)
  - [v0.5 (initial release)](#v05-initial-release)

## 1.8.0
### Features
#### CUDA Graphs
 - Enable CUDA Graphs for TRT (#19184)
 - CUDA graphs support (#19142)
 - Update cudnn version. (#19375)
#### CUDA 11 Support
 - Update CUB and include it only for CUDA < 11 #18799' (#18975)
 - Add new CI pipeline for building and testing with cuda 11.0. (#19149)
 - Enable CUDA 11.0 on nightly development builds (#19314)
#### TensorRT
 - TensorRT: add int8 with calibration (#19011)
 - Add TRT verbose mode (#19100)
 - Backporting TensorRT-Gluon Partition API (and TensorRT 7 support) (#18916)
 - Backport TRT test update #19296 (#19298)
#### OneDNN
 - Upgrade to oneDNN v1.6.3 (#19153) (#19161)
 - Update oneDNN to official v1.6 release (#18867) (#18867)
 - Upgrade to oneDNN v1.6 (#18822)
 - bumped version to v1.6.5 (#19437)
 - Upgrade to oneDNN v1.7 (#19560)
#### IntGemm
 - Backport of intgemm #17559 (#19099)
 - Change intgemm to a submodule instead of fetch. (#19406)
#### Subgraph API
 - Backport Fix for duplicate subgraph inputs/outputs (#16131) (#19112)
#### Extensions
 - Backport #19103 (#19117)
 - Backporting #19016 (#19069)
 - Backport: Change Partition API's options_map to std::unordered_map #18929 (#18964)
 - Backporting #18779 to v1.x (#18894)
 - Backport extension bug fixes to v1.8.x (#19469) (#19504)
 - fix for MX_ERROR_MSG namespace (#19756)
#### ONNX
 - Update onnx support to work with onnx 1.7.0 with most CV models (#19017)
#### Large Tensor
 - Fix linalg_potri and linalg_potrf operators for large tensor. (#18752)
 - Add forward, backward test for linalg.gemm2 (#18784)
 - Add large matrix tests for linalg ops: det, inverse, trsm, trmm (#18744)
 - Add Large Tensor Test for linalg_syrk (#18782)
 - Add Large Dim Checks for linalg Operators (#18816)
 - Add forward & backward linalg.gemm test for large size (#18825)
 - Adding error message when attempting to use Large tensor with linalg_syevd (#18807)
#### Website Improvements
 - v1.8 website patch (#19212)
 - Automate website artifacts uploading (#19244)
#### Documentation
 - Fix mxnet.test_utils.check_numeric_gradient documentation (#19060)
 - Update windows_setup.md (#18874)
#### License
 - Stop packaging GPL libquadmath.so (#19055)
 - Remove mention of nightly in pypi (#18635) (#18884)
 - Mkldnn header fix v1x for nightly binaries (#18797)
 - Update LICENSE for all submodules. (#19440)
 - LICENSE update (#19443)
 - Update LICENSE (#19704) (#19707)
#### CI Improvements
 - Upgrade unix gpu toolchain (#18186) (#18785)
 - Fix CI in v1.x branch (#18907)
 - Remove extra --build-arg causing docker command to fail. (#19412)
 - Fix CI builds failing due to invalid GPG keys. (#19377) (#19388)
#### Bug Fixes
 - Backport #19656 - fix R builds (#19658)
 - remove cleanup on side threads (#19557)
 - Don't use namespace for pow() function, since it is built into cuda math library, and cast the second argument so it will find an acceptable form. (#19533)
 - Remove temporary fix for RNN (#19451)
 - backport #19393 to v1.8.x (#19398)
 - Fix SoftReLU fused operator numerical stability (#17849) (#19390)
 - Temporary fix for RNN with oneDNN seg faults/core dumps (#19308)
 - Fix MKLDNN BatchNorm with even number of channels (#19150) #19299 #19425 (#19428)
 - Relaxing type requirements for broadcast_like (#17977) (#19448)
 - Backporting: Fixed setting attributes in reviewSubgraph (#19278)
 - Include oneDNN gemm fix (#19251)
 - Fix for breaking change introduced in #17123 when batch_axis=0 (#19283)
 - Backport PR #19272 to v1.8.x (#19273)
 - Backport PRs in v1.7.x missing from v1.x to v1.8.x (#19262)
 - Delete executor before reallocating it memory (#19222)
 - Nightly Large Tensor test cherrypicks (#19194) (#19215)
 - Tweeking syntax to be closer to other tests (#19186) (#19206)
 - ElementWiseSum fix for oneDNN (#18777) (#19200)
 - Fix flaky intgemm test in v1.8.x too (#19204)
 - Revert "Fix memory leaks in Gluon (#18328) (#18359)" (#19181)
 - Improve environment variable handling in unittests (#18424) (#19173)
 - Backport Unittest tolerance handling improvements (#18694). Also test seeding (#18762). (#19148)
 - Fix the error of gradient of np.pad (#19044) (#19167)
 - Backport Add cmake flag USE_FATBIN_COMPRESSION, ON by default (#19123) (#19158)
 - SymbolBlock.imports ignore_extra & allow_missing (#19156)
 - Fix race condition in NaiveEngine::PushAsync (#19108) (#19122)
 - Empty list cannot be cleared issue fixed. (#14882)
 - Update base_module.py (#19096)
 - Fix block.export (#17970) (#19075)
 - Support for fp16 in SpM x DnsM on GPU (#18930) (#19074)
 - Backport of Fix LeakyRelu behaviour on empty input (#18934) (#19009)
 - Get rid of monkey patching in LossScaler overflow handling (#18959) (#18973)
 - Remove upper bound (#18857) (#18910)
 - Fix gelu to use erf based algorithm (#18827) (#18946)
 - Cherry-pick #18635 to v1.7.x (#18935) (#18945)
 - Backporting backward inference from 2.x #18348 and #18378 (#18895)
 - Backport Invoke mkldnn and cudnn BatchNorm when axis != 1 to v1.7.x (#18676) (#18890)
 - Bump version to 1.8.0 (#18899)
 - Fixing ONNX spatial export for batchnorm (#17711) (#18846)
 - Fix softmax, logsoftmax failed on empty ndarray (#18602) (#18708)
 - Add unit tests for potri and potrf backward and check output shape in unit tests. (#18803)
 - Add syrk test shape check (#18812)
 - Back port optimization to broadcast_axis to MXNet1.x (#18773)
 - Fix crash when accessing already destructed static variables (#18768) (#18778)
 - Cherrypick #18677 #18713 (#18742)

## 1.7.0

### New features
#### MXNet Extensions: custom operators, partitioning, and graph passes

Adds support for extending MXNet with custom operators, partitioning strategies, and graph passes. All implemented in a library easily compiled separately from the MXNet codebase, and dynamically loaded at runtime into any prebuilt installation of MXNet.

 - fix for number of inputs/outputs for backward custom ops (#17069)
 - Enhancements for custom subgraph op (#17194)
 - Disable flaky test_custom_op_fork (#17481)
 - fix custom op makefile (#17516)
 - Update CustomOp doc with changes for GPU support (#17486)
 - [WIP] MXNet Extensions enhancements (#17885) (#18128)
 - Dynamic subgraph property (#17034)
 - Dynamic subgraph property doc (#17585)
 - [1.7] Backport MXNet Extension PRs (#17623, #17569, #17762) #18063 (#18069)

#### OpPerf utility enabled in the binary distribution
 - [OpPerf] Add Neural network loss ops (#17482)
 - [OpPerf] Fixes the issue when you pass NDArray to run_perf_test (#17508)
 - [OpPerf] Fix markdown for native profile and add profile param in function desc (#17494)
 - [OpPerf] Add Indexing ops (#16253)
 - [OpPerf] Implement remaining random sampling ops (#17502)
 - [OpPerf] Implement remaining GEMM ops (#17501)
 - [OpPerf] Implement all linalg ops (#17528)
 - [OpPerf] Fixed native output ordering, added warmup & runs command line args (#17571)
 - [OpPerf] Add norm, cast ops, remaining optimizer ops (#17542)
 - [OpPerf] Fixed Python profiler bug (#17642)

#### MKL-DNN
##### MKL-DNN as the default CPU backend in binary distribution
##### Branding change to DNNL
 - Upgrade MKL-DNN dependency to v1.1 (#16823)

##### Support bfloat16 datatype
 - Add bfloat16 floating-point format support based on AMP  (#17265)

#### New operators
 - [New Op] Add deformable conv v2 (#16341)
 - Add MXNet Ops for fast multihead attention (#16408)
 - Support boolean elemwise/broadcast binary add, multiply and true_divide (#16728)
 - add gammaln, erf, erfinv (#16811)
 - add aligned roi introduced in Detectron2 (#16619)
 - Implement atleast_1d/2d/3d (#17099)
 - Interleaved MHA for CPU path (#17138)
 - Lamb optimizer update (#16715)
 - Quantized Embedding (#16691)
 - Add gelu fuse ops (#18082) (#18092)

### Feature improvements
#### Numpy compatible interface(experimental)
 - [NumPy] NumPy support for linalg.inv (#16730)
 - add numpy op nan_to_num (#16717)
 - [Numpy] Add sampling method for bernoulli (#16638)
 - Fix numpy-compatible mean output type for integer inputs (#16792)
 - [Numpy] Fix collect_params().zero_grad() in gluon numpy interface (#16716)
 - [Numpy][Operator] 'where' Implementation in MXNet (#16829)
 - [Numpy] Random.normal() with backward (#16330)
 - Add OP diag [numpy] (#16786)
 - Mixed precison binary op backward (use in) for numpy (#16791)
 - add numpy op diagflat [numpy] (#16813)
 - add op bitwise_or [numpy] (#16801)
 - [Numpy] Implementation npx.{sample}_n (#16876)
 - [Numpy] Add NumPy support for np.linalg.det and np.linalg.slogdet (#16800)
 - Op Unravel_index PR [Numpy] (#16862)
 - [Numpy] Fix imperative basic indexing in numpy (#16902)
 - [Numpy] Basic indexing in symbolic interface of DeepNumpy (#16621)
 - [Numpy] add op full_like, c++ impl, fix zeros_like, ones_like type inference (#16804)
 - [Numpy] Implement numpy operator 'average' (#16720)
 - [Bugfix] [Numpy] Add `kAddTo` and kNullOp to Transpose (#16979)
 - set rtol = 1e-2 and atol = 1e-4 when dtype == np.float32 in test_numpy_op.py:test_np_linalg_solve (#17025)
 - Op_Diagonal [Numpy] (#16989)
 - numpy bincount (#16965)
 - [numpy] add op bitwise_not (#16947)
 - [Numpy ]Modify np.random.shuffle to enable inplace by default (#17133)
 - [numpy] fix argsort typo (#17150)
 - [numpy] add op round (#17175)
 - [numpy]Add op delete (#17023)
 - [numpy] add op flipud, fliplr (#17192)
 - [CI] Re-enable testing with numpy 1.18 (#17200)
 - [Numpy] Add broadcast_to scalar case (#17233)
 - [Numpy] Random.gamma() implemented (#16152)
 - [Numpy] add row_stack (=vstack) (#17171)
 - [Numpy] Add infra for performing constraint check (#17272)
 - porting numpy-compatible hstack to master and add dstack for interoperability (#17030)
 - adding asnumpy() to output of gather(implicitly called) to fix gather test in large vector and tensor tests (#17290)
 - [numpy] add op random.exponential (#17280)
 - [NumPy] Add NumPy support for norm (#17014)
 - [numpy]add op random.lognormal  (#17415)
 - Add numpy random weibull operator (#17505)
 - [numpy] Add np.random.pareto and np.random.power (#17517)
 - [Numpy] Add sort op (#17393)
 - [numpy]implement exponential backward (#17401)
 - [Numpy] Where operator scalar version (#17249)
 - [numpy] add op matmul (#16990)
 - [numpy]add op random.logistic, random.gumbel (#17302)
 - [numpy][Do Not Review]add op insert (#16865)
 - [numpy] add op random.rayleigh (#17541)
 - [numpy] add fallback ops (#17609)
 - [numpy] add op pad (#17328)
 - [numpy] add op fabs, sometrue, round_ (#17619)
 - Add arange_like to npx (#16883)
 - try to move shape_array to npx (#16897)
 - support np.argsort (#16949)
 - np.broadcast_to extension (#17358)
 - support bitwise_and (#16861)
 - fix np.argmax/argmin output data type (#17476)
 - add op random.beta (#17390)
 - add op isnan isinf (#17535)
 - array_split pr (#17032)
 - Mixed data type binary ops (#16699)
 - randn implemented (#17141)
 - refactor and reduce float types for some functions, also add bitwise_xor (#16827)
 - any/all (#17087)
 - amax (#17176)
 - fix format (#17100)
 - add op empty_like, add nan_to_num to dispatch (#17169)
 - handle array_like fill_value for np.full; add unit test coverage (#17245)
 - add np.amin (#17538)
 - add npx.gather_nd (#17477)
 - add np.random.chisquare (#17524)
 - add polyval (#17416)
 - add isposinf isneginf isfinite (#17563)
 - Support broadcast assign for `npi_boolean_mask_assign_tensor` (#17131)
 - Implement Weibull backward (#17590)
 - support np.dsplit, fix some error msgs and corner cases for hsplit and vsplit, add interoperability tests for h/v/dsplit (#17478)
 - add np.product (#17489)
 - Implement np.random.pareto backward (#17607)
 - add np.ediff1d (#17624)
 - more support for boolean indexing and assign (#18352)
 - Fix einsum gradient (#18482)
 - [v1.7.x] Backport PRs of numpy features (#18653)
 - [v1.7.x] backport mixed type binary ops to v1.7.x (#18649)
 - revise activations (#18700)

#### Large tensor support
 - [Large Tensor] Add support to Random Sample & Pdf ops (#17445)
 - [Large Tensor] Add LT support for NN optimizers and 1 activation function (#17444)
 - [Large Tensor] Fixed SoftmaxActivation op (#17634)
 - [Large Tensor] Fixed col2im op (#17622)
 - [Large Tensor] Fixed Spatial Transformer op (#17617)
 - [Large Tensor] Fix ravel_multi_index op (#17644)
 - Sparse int64 Large tensor support (#16898)
 - Re-Enabling Large Tensor Nightly on GPU (#16164)
 - enabling build stage gpu_int64 to enable large tensor nightly runs (#17546)
 - [Large Tensor] Fixed Embedding op (#17599)

#### MKL-DNN enhancement
 - MKLDNN FC : Add error info when mkldnn fc bias dimension is wrong (#16692)
 - [MKLDNN] support mkldnn gelu (#16710)
 - [MKLDNN] Fix int8 convolution/fc bias overflow (#16734)
 - [MKLDNN] use dim_t instead of int in slice/transpose operators (#16737)
 - Mkldnn fullyConnect bwd bug fix (#16890)
 - Revert Mkldnn fullyConnect bwd bug fix (#16890) (#16907)
 - [MKLDNN] Use MKLDNNRun (#16772)
 - [MKLDNN] mkldnn RNN operator enhancement (#17075)
 - [MKLDNN] enable MaxPooling with full pooling convention (#16860)
 - update mkldnn to v1.1.2 (#17165)
 - improve mkldnn doc (#17198)
 - [MKLDNN] Fix _copyto  (#17173)
 - [MKLDNN] Support channel wise quantization for FullyConnected (#17187)
 - fixed seed for mkldnn test (#17386)
 - add mkldnn softmax backward  (#17170)
 - cmake: copy dnnl headers to include/mkldnn (#17647)
 - [mkldnn]Mkldnn bn opt backport from master to 1.7x (#18009)
 - [v1.x] Update 3rdparty/mkldnn remote URL and pin to v1.3 (#17972) (#18033)
 - [v1.x] backport #17900 [MKLDNN] support using any format in pooling backward (#18067)
 - Static link MKL-DNN library (#16731)
 - Add large tensor nightly tests for MKL-DNN operators (#16184)
 -  [MKL-DNN] Enable and Optimization for s8 eltwise_add (#16931)
 - [MKL-DNN] Enhance Quantization Method (#17161)
 - Static Build and CD for mxnet-cu102/mxnet-cu102mkl (#17074)
 - MKL-DNN RNN backward path enhancement (#17183)
 - cmake: check USE_OPENMP and pass proper MKL-DNN build flags (#17356)
 - update mkl to 2020.0 (#17355)
 - Enable MKL-DNN by default in pip packages (#16899)
 - Enable MKL-DNN FullyConnected backward (#17318)
 - Softmax primitive cache and in-place computation (#17152)
 - boolean_mask_assign with start_axis (#16886)
 - use identity_with_cast (#16913)
 - change error tolerance for bf16 bn (#18110)
 - [v1.x] Backport #17689 and #17884 to v1.x branch (#18064)
 - refactor codes and add an option to skip/check weight's version to reduce overhead (#17707) (#18039)
 - [v1.x] Backport #17702 and #17872 to v1.x branch (#18038)

#### TensorRT integration
 - Update TensorRT tutorial to build-from-source. (#14860)
 - Minor fix, use RAII for TensorRT builder and network object (#17189)

#### Quantization
 - Add silent option to quantization script (#17094)

#### Profiler
 - Implemented final two binary ops, added default params for functionality (#17407)
 - Implement remaining nn_activation ops in opperf (#17475)
 - Implement all miscellaneous ops (#17511)
 - Implement remaining nn_basic ops in opperf (#17456)

#### ONNX
 - Fix memory leak reported by ASAN in NNVM to ONNX conversion (#15516)
 - ONNX export: Gather (#15995)
 - ONNX export: Slice op - Handle None value for ends (#14942)

#### New models
 - [Model] Implement Neural Collaborative Filtering with MXNet (#16689)
 - Further optimization for NCF model (#17148)
 - HMM Model (#17120)

#### Operator improvements
 - Faster GPU NMS operator (#16542)
 - [MXNET-1421] Added (CuDNN)BatchNorm operator to the list of mirrored operators (#16022)
 - dynamic custom operator support (#15921)
 - Multi Precision Lamb Update operator (#16885)
 - Add im2col and col2im operator (#16502)
 - Quantized Elemwise Mul Operator (#17147)
 - Enhancements for MXTensor for custom operators (#17204)
 - Enabling large tensor support for binary broadcast operators (#16755)
 - Fix operators lying about their number of inputs (#17049)
 - [WIP] Fallback mechanism for mx.np operators (#16923)
 - Dynamic custom operator GPU support (#17270)
 - Fix flaky - test_operator_gpu.test_np_insert (#17620)
 - MXNet FFI for Operator Imperative Invocation (#17510)
 - [MXNET-978] Higher Order Gradient Support `logp1`, `expm1`, `square`. (#15416)
 - [MXNET-978] Higher Order Gradient Support `arcsin`, `arccos`. (#15515)
 - [MXNET-978] Higher Order Gradient Support `rsqrt`, `rcbrt`. (#15476)
 - gather_nd: check bound and wrap negative indices (#17208)
 - Remove dilation restriction for conv3d (#17491)
 - Fix storage type infer of softmax backward (#17576)
 - Fix and optimize handling of vectorized memory accesses (#17767) (#18113)
 - Cherry-pick of #17995 and #17937 to 1.x branch (#18041)
 - No tensor cores for fp32 interleaved attention, remove div by 8 restriction (#17994) (#18085)
 - GPU gemms true fp16 (#17466) (#18023)
 - Add support for boolean inputs to FusedOp (#16796)

#### Bug fixes
 - [BUG FIX] Always preserve batch dimension in batches returned from dataloader (#16233)
 - Fix SliceChannel Type inference (#16748)
 - change _generate_op_module_signature get_module_file open with encoding=utf-8,it fix some encode error in Chinese windows system. (#16738)
 - Fix rtrue_divide grad (#16769)
 - fix inv test flakiness using random matrices generated by SVD (#16782)
 - [MXNET-1426] Fix the wrong result of sum, mean, argmin, argmax when inputs contain inf or nan (#16234)
 - Fix (#16781)
 - fix expand_dims fall back when input's ndim is 0 (#16837)
 - [fix] missing input log higher order. (#15331)
 - Fix IndentationError in setup.py (#16857)
 - Fix a few np issues (#16849)
 - Fix InferAttr/InferShapeAttr not calling inference for all nodes in a graph (#16836)
 - fix for enable model parallelism for non-fp32 data (#16683)
 - Fix NDArrayIter iteration bug when last_batch_handle='pad' (#16166)
 - Fix crashing on Windows in ObjectPool ~ctor (#16941)
 - Fix NDArrayIter cant pad when size is large (#17001)
 - fix axis=-1 bug (#17016)
 - Fix CUDNN detection for CMake build (#17019)
 - Fix omp assert issue (#17039)
 - mshadow: fix vector access (#17021)
 - [BUGFIX] Fix race condition in kvstore.pushpull (#17007)
 - [BUGFIX] Fix trainer param order (#17068)
 - [BugFix] fix filter channel calculation in ModulatedDeformableConvV2 (#17070)
 - Fix reshape interoperability test (#17155)
 - fix norm sparse fallback (#17149)
 - fix py27 quantization (#17153)
 - fix int8 add ut (#17166)
 - Fix and clean up Ubuntu build from source instructions (#17229)
 - fix lstm layer with projection save params (#17266)
 - Fix rendering of ubuntu_setup.md codeblocks (#17294)
 - Fix #17267, add expected and got datatype for concat error msgs (#17271)
 - [BUGFIX] fix model zoo parallel download (#17372)
 - fix use int8, uint8, int32, int64 (#17188)
 - [Fix] Add ctx to the original ndarray and revise the usage of context to ctx (#16819)
 - Fix ndarray indexing bug (#16895)
 - fix requantize flaky test (#16709)
 - Initial checkin (#16856)
 - Fix flakey test_ndarray.py:test_reduce (#17312)
 - fix flaky test: boolean index and fix bugs (#17222)
 - Fix IOT Devices section of Get Started page (#17326)
 - add logic for no batch size while getting data arrays from executors (#17772) (#18122)
 - Fix reverse shape inference in LayerNorm (#17683)
 - fix full and full_like when input is boolean (#17668)
 - Fix MBCC inference (#17660)
 - Additional fix for vector access. (#17230)
 - Cherrypick Fix nightly large_vector test caused by incorrect with_seed path (#18178) (#18220)
 - [1.7] Pass args fix3 (#18237)
 - fixing batch_norm and layer_norm for large tensors (#17805) (#18261)
 - [1.7.x] Backport of LSTM and GRU fix (#17898) and RNN op (#17632) (#18316)
 - [v1.7.x] backport #18500 - [Bug Fixed] Fix batch norm when grad_req is `add` (#18517)
 - Fix the monitor_callback invalid issue during calibration with variable input shapes (#18632) (#18703)

### Front end API
 - Fix the problem in printing feature in c++ API examples : feature_extract (#15686)
 - updating MXNet version to 1.6.0 in base.h for C APIs (#16905)
 - [API] unified API for custom kvstores (#17010)
 - fix parameter names in the estimator api (#17051)
 - adding docs for 64bit C APIs of large tensor (#17309)
 - Add API docs to INT64 APIs (#16617)

#### Gluon
 - [Quantization] Enhance gluon quantization API (#16695)
 - [Gluon] Improve estimator usability and fix logging logic (#16810)
 - Fix test_gluon.py:test_sync_batchnorm when number of GPUS > 4 (#16834)
 - [Gluon] Update contrib.Estimator LoggingHandler to support logging per batch interval (#16922)
 - Include eval_net the validation model in the gluon estimator api (#16957)
 - Fix Gluon Estimator nightly test (#17042)
 - [MXNET-1431] Multiple channel support in Gluon PReLU (#16262)
 - Fix gluon.Trainer regression if no kvstore is used with sparse gradients (#17199)
 - refactor gluon.utils.split_data() following np.array_split() (#17123)
 - Add RandomApply in gluon's transforms (#17242)
 - Partitioning Gluon HybridBlocks (#15969)
 - Random rotation (#16794)
 - bump up atol for gradient check (#16843)
 - Extend estimator.evaluate() to support event handlers (#16971)
 - [MXNET-1438] Adding SDML loss function (#17298)

#### Symbol
 - Add unoptimized symbol to executor for sharing (#16798)
 - Enforces NDArray type in get_symbol (#16871)
 - Fix #17164 symbolblock with BatchNorm inside during cast to fp16 (#17212)
 - autograd video and image link fixes and removing symbol tutorials (#17227)
 - Fix CosineEmbeddingLoss in when symbol API is used (#17308)
 - Fix Horovod build error due to missing exported symbols (#17348)
 - Update symbol.py (#17408)
 - update symbol to json (#16948)

### Language Bindings
#### Python
 - Python 2 compatibility fix in base.py
 - adding stacktrace in Jenkinsfile_utils.groovy to inspect Python2 failure cause in CI (#17065)
 - Fix image display in python autograd tutorial (#17243)
 - Fix Python 3 compatibility in example/speech_recognition (#17354)
 - Stop testing Python 2 on CI (#15990)
 - Docs: Python tutorials doc fixes (#17435)
 - pin python dependencies (#17556)
 - Python 2 cleanup (#17583)

#### C/C++
 - Simplify C++ flags (#17413)

#### R
 - fix R docs (#16733)
 - [R package] Make R package compilation support opencv 4.0 (#16934)
 - Support R-package with cmake build and fix installation instructions (#17228)
 - Fix R-package/src/Makevars for OpenCV 4 (#17404)
 - Fix typo in Install the MXNet Package for R (#17340)

#### Clojure

#### Julia
 - [MXNET-1440] julia: porting `current_context` (#17142)
 - julia: porting `context.empty_cache` (#17172)
 - pin Markdown version to 3.1 in Julia doc build (#17549)

#### Perl
 - [Perl] - ndarray operator overloading enhancements (#16779)
 - MXNET-1447 [Perl] Runtime features and large tensor support. (#17610)

#### Scala
 - Fix scala publish & nvidia-docker cublas issue (#16968)
 - Fix publishing scala gpu with cpu instance (#16987)
 - swap wget to curl in Scala scripts (#17041)
 - [Scala/Java] Remove unnecessary data slicing (#17544)
 - quantile_scalar (#17572)
 - Fix get_started scala gpu (#17434)
 - Fix MBCC & scala publish pipeline (#17643)
 - Bump up additional scala 1.x branch to 1.7.0 (#17765)

### Performance improvements
 - Build.py improvement (#16976)
 - Improvements to config.cmake (#17639)
 - [Done] BilinearResize2D optimized (#16292)
 - Speed fused_op compilation by caching ptx and jit-compiled functions (#16783)
 - Improve the speed of the pointwise fusion graph pass (#17114)
 - broadcast_axis optimization (#17091)
 - Optimize AddTakeGrad Tensor Sum (#17906) (#18045)

### Example and tutorials
 - Add CustomOp tutorial doc (#17241)
 - Correct the grammar in 1-ndarray tutorial (#17513)

### Website and documentation
 - Website edits (#17050)
 - [Website 2.0] Nightly Build for v1.x (#17956)
 - [docs] Fix runtime feature detection documentation (#16746)
 - Adding user guidelines for using MXNet built with Large Tensor Support (#16894)
 - fix typo and doc (#16921)
 - large tensor faq doc fix (#16953)
 - [DOC] Add a few tips for running horovod (#17235)
 - Update NOTICE to fix copyright years (#17330)
 - [DOC] Fix tutorial link, and better error msg (#17057)
 - doc fix for argmax & argmin (#17604)

### CI/CD
 - support mixed-precision true_divide (#16711)
 - Try to fix CI (#16908)
 - mixed precision for power (#16859)
 - Fix desired precision for test_ndarray.py:test_reduce (#16992)
 - [reproducibility] multi_sum_sq review, AtomicAdd removal (#17002)
 - fix precision problem in linalg_solve, linalg_tensorinv, linalg_cholesky op test (#16981)
 - grouping large array tests based on type and updating nightly CI function (#17305)
 - [LICENSE] fix cpp predcit license (#17377)
 - [CI] Fix static build pipeline (#17474)
 - skipping tests that cannot fit in nightly CI machine corrected imports (#17450)
 - Update Windows CI scripts to use syntax compatible with Win 2019 server powershell. (#17526)
 - Fix Non-ASCII character in docstring (#17600)
 - [CI] Follow redirects when downloading apache-maven-3.3.9-bin.tar.gz (#17608)
 - [CI] Upgrade sphinx and autodocsumm (#17594)
 - Reduce load on CI due to excessive log flood (#17629)
 - Enable users to specify BLAS (#17648)
 - [CI] Add AMI id to instance info on builds (#17649)
 - [v1.7.x] Backport staggered CI builds (#17999 & #18119) (#18142)
 - [v1.7.x] Backport #17177 to 1.7.x (Fix incorrect calculation results when the C locale is set to a locale that uses commas as the decimal separator) (#18147)
 - Fix formatting and typos in CD README.md (#16703)
 - [CD] dynamic libmxet pipeline fix + small fixes (#16966)
 - [CD] enable s3 publish for nightly builds in cd (#17112)
 - [CD] fix CD pipeline (#17259)
 - [CD] update publish path (#17453)
 - fix CD and remove leftover from #15990 (#17551)
 - Fix nightly build (#16773)
 - Update pypi_publish.py to disable nighlty build upload to Pypi (#17082)
 - [v1.7.x] update jetson dockerfile to support CUDA 10.0 (#18339)
 - Remove manually created symbolic link to ninja-build (#18437) (#18456)
 - Increase staggered build timeout to 180 min (#18568) (#18585)

### License
 - Don't relicense FindCUDAToolkit.cmake (#17334)
 - fix license and copyright issues (#17364)
 - Update ps-lite LICENSE (#17351)
 - remove unused file with license issue (#17371)
 - Update LICENSE for fonts (#17365)
 - license np_einsum file under bsd (#17367)
 - Update Apache License for mshadow (#18109) (#18134)
 - Julia: remove downloading of the non-ASF binary build (#18489) (#18502)
 - Add missing license header for md files (#18541)
 - [v1.7.x]License checker enhancement (#18478)

### Miscellaneous changes
 - Link fixes4 (#16764)
 - Refactoring names for mxnet version of nnvm to avoid conflicting with the original tvm/nnvm. (#15303)
 - minor typo fix (#17008)
 - Add micro averaging strategy to pearsonr metric (#16878)
 - introduce  gradient update handler to the  base estimator (#16900)
 - fix latency calculation and print issue (#17217)
 - add inference benchmark script (#16978)
 - change the wording and log level to be more in line with the general use (#16626)
 - Updated logos. (#16719)
 - Pinning rvm version to satisfy Jekyll build (#18016)
 - Workaround gnu_tls handshake error on Ubuntu 14.04 Nvidia Docker (#18044)

## 1.6.0

### Deprecation of Python 2

MXNet community [voted](https://lists.apache.org/thread.html/r3a2db0f22a1680cc56804191446fef2289595798ca19fd17de1ff03e%40%3Cdev.mxnet.apache.org%3E) to no longer support Python 2 in future releases of MXNet. Therefore, MXNet 1.6 release is going to be the last MXNet release to support Python 2.

### New features

#### NumPy compatible interface and using TVM to generate operators

NumPy has long been established as the standard math library in Python, the most prevalent language for the deep learning community. With this library as the cornerstone, there are now the largest ecosystem and community for scientific computing. The popularity of NumPy comes from its flexibility and generality.

In #14253, the MXNet community reached consensus on moving towards a NumPy-compatible programing experience and committed to a major endeavor on providing NumPy compatible operators.

The primary goal of the projects below is to provide the equivalent usability and expressiveness of NumPy in MXNet to facilitate Deep Learning model development, which not only helps existing deep learning practitioners but also provides people in the existing NumPy community with a shortcut for getting started in Deep Learning. The efforts towards this goal would also help a secondary goal, which is to enable the existing NumPy ecosystem to utilize GPUs and accelerators to speed up large scale computation.

 - Infra to use tvm write op kernels (#15550)
 - fix boolean_mask for 0-size output (#15731)
 - fix tvm cmake (#15781)
 - Numpy-compatible Infra (#15581)
 - [MXNET-1206] Support NDArray indexing with None and Ellipsis (#13143)
 - numpy-compatible sum (#15810)
 - [Numpy] Numpy compatible slicing (#15798)
 - Numpy Tensordot and Dot Operator (#15820)
 - numpy linspace (#15852)
 - tvm infra for op attrs (#15854)
 - Port several np ops to master (#15867)
 - numpy-compatible split upstream (#15841)
 - Numpy-compatible concatenate upstream (#15894)
 - Numpy-compatible stack upstream (#15842)
 - [Numpy] Numpy behavior random.uniform() (#15858)
 - Tvm broadcast backward (#15938)
 - np elemwise unary ops upstream (#15831)
 - [Numpy] random.randint() implemented (#15956)
 - Refines NDArray indexing and adds numpy ndarray indexing [READY FOR REVIEW] (#15942)
 - Port ops from np branch (#16018)
 - numpy-compatible cumsum upstream (#15924)
 - NumPy-compatible infrastructure on Gluon (#16024)
 - [OP] Support range as advanced index for ndarrays (#16047)
 - Numpy compatible max min (#16046)
 - NumPy-compatible Mean, Std and Var (#16014)
 - Add fluent methods mean, std, var for ndarray (#16077)
 - numpy multinomial op (#15878)
 - add numpy operator remainder (#16080)
 - [Numpy] Random.choice implemented (#16089)
 - Fix sample.normal shape inference
 - Numpy add numpy op indices (#15837)
 - [Numpy] Numpy copysign (#15851)
 - numpy operator ravel, derive from reshape (#16016)
 - Add __array_function__
 - Improved error mesages
 - Fix np.choice
 - add exception check for numpy reshape (#16180)
 - [Numpy] Numpy behavior normal distribution (#16109)
 - fix multinomial bug on gpu (#16204)
 - [Numpy] Differentiable svd (#15795)
 - add epsilon to sum(pvalue) upperbound (#16211)
 - np compatible vstack (#15850)
 - Numpy add numpy op roll (#15902)
 - add numpy compatible trace (#16008)
 - add numpy op hanning, hamming, blackman (#15815)
 - [Numpy]flip (#15819)
 - numpy operator around (#16126)
 - numpy operator arctan2 (#15890)
 - numpy operator nonzero (#15838)
 - numpy operator hypot (#15901)
 - tvm numpy operator deg2rad && rad2deg (#16015)
 - numpy op unique
 - try to fix bug
 - fix memory bug and disable some test
 - fix according to review
 - Numpy operators: `lcm`, `tril`, `identity` and `take` (#16264)
 - [numpy] Cosmetic improvement on mxnet.numpy builtin op signature in documentation (#16305)
 - Disable Pylint false error in numpy_op_signature  (#16370)
 - boolean_mask_assign operator for future boolean indexing (#16361)
 - Implements ldexp. (#15845)
 - Numpy Operators: Inner, Outer, vdot (#15846)
 - Numpy det and slogdet operators (#15861)
 - Fix random op signature
 - fix choice signature
 - add raise test for shape
 - Add boolean ndarray (#15940)
 - global numpy shape flag (#16335)
 - numpy-compatible histogram (#16266)
 - [Numpy] Numpy compatible dstack (#15871)
 - numpy eye op (#16132)
 - Numpy compatible vsplit; minor changes to split (#15983)
 - add numpy op logspace (#15825)
 - add numpy op bitwise_xor, hsplit, moveaxis, rot90 (#16257)
 - Fix optimizer bug for np attribute (#16494)
 - Tests of NumPy interoperability (#16469)
 - improve unary and binary operator handling and refactor tests (#16423)
 - [DOC] Fix numpy op doc  (#16504)
 - [Numpy] More numpy dispatch tests (#16426)
 - [Numpy] einsum (#15911)
 - Add test pipeline for USE_TVM_OP=OFF on Unix (#16450)
 - Numpy dispatch test of ...... (#16422)
 - setup and concatenate, copy, expand_dims, expm1 (#16493)
 - add sum for boolean type in mainline (#16436)
 - [Numpy] SVD outputs tuple (#16530)
 - numpy op doc: max, min, prod (#16506)
 - add interface for rand
 - Fix numpy bugs (#16537)
 - pickler override for np ndarrays (#16561)
 - [numpy]op test in new pattern (#16556)
 - Enforce adding documentation for builtin numpy operators (#16575)
 - [Numpy] Support N_D(N>=3) batch_dot (#16586)
 - [Numpy] Loading numpy-incompatible NDArray in numpy-compatible mode (#16597)
 - Fix index overflow bug in einsum (#16589)
 - add npx reshape (#16640)
 - add type switch to weight tensor (#16543)
 - numpy doc enhancement (#16637)
 - Infra for tvm op runtime dispatch (#16100)
 - [NumPy][Operator] NumPy operator `may_share_memory` and `shares_memory` (#16533)
 - [Numpy] Numpy operator diff (#15906)
 - Miscellaneous fix for several numpy issues (#16664)
 - [Numpy] implement np.column_stack (#16594)
 - [numpy] add numpy operator : append (#16564)
 - Backport of #16711, #16737, #16408 to 1.6 branch (#16763)
 - Backport to 1.6 (#16773, #16781, #16783, #16716, #16699, #16728, #16769, #16792) (#16832)
 - [Backport][v1.6.x] Fix the wrong result of sum, mean, argmin, argmax when inputs contain inf or nan (#16884)
 - Backport of #16827, #16791 and #16888 to 1.6 branch (#16901)
 - port shape op to 1.6.x (#16912)
 - [Numpy] Fix imperative basic indexing in numpy (#16902) (#16919)
 - Backport #16895, #16922, #16878, #16979 and #16900 to 1.6 (#17029)


#### Graph optimizations

##### Pointwise fusion for GPU

DL models, besides compute intensive operations like convolutions and fully connected layers, feature a lot of simple pointwise (aka elementwise) operations (like elementwise addition etc.). Performance of those operations is fully memory bandwidth bound and so limit speedups from newer GPU hardware, which typically has high compute/memory bandwidth ratio. When multiple of such operations are chained one after another, it results in a series of unnecessary stores and loads as well as potential increased memory usage to store the intermediate results. Pointwise fusion helps in alleviating those problems by just-in-time generation of fused operators, which do not store intermediate results in memory, resulting in performance and memory usage improvements.

 - Pointwise fusion for GPU (#15167)
 - Backport #16798, #16836 and #16838 to 1.6 (#16874)
 - Add support for boolean inputs to FusedOp (#16796) (#16892)
 - Workaround problem with fusion in CUDA 9 (#17028) (#17035)

##### Eliminate common subexpressions

 - Eliminate common expressions (#15657)

##### Default MKLDNN Subgraph fusion

 - [MKLDNN] Enable subgraph backend mkldnn by default. (#15518)

#### New operators

 - [OP] Add a new arange_like operator to contrib (#15400)
 - PDF operators for each distribution for which we have a random sampler (plus also the PDF of the Dirichlet).  Supports probabilities and log-probabilities, as well as gradients. (#14617)
 - Group Normalization (#14959)
 - Add RROIAlign (#16017)
 - Add fast implementation of LARS (#16122)
 - Round and sign straight-through-estimators C operators. (#16373)
 - New ops for RCNN + old ops improvements for RCNN (#16215)
 - Comparison ops implemented using mshadow (#16414)
 - Add mask target generator operator for Mask-RCNN (#16268)
 - Move MRCNNMaskTarget op to contrib (#16486)
 - Mxnet allclose (#14443)
 - Aggregated adamw update (#16398)
 - Make mrcnn_mask_target arg mask_size a 2d tuple (#16567)
 - Dgl ops 2 (#16416)
 - Lamb optimizer update (#16715)
 - [OP] changing data type of 't' to int in lamb_update_phase1 (#16903)
 - Multi Precision Lamb Update operator (#16885)
 - Interleaved MHA for CPU path (#17138) (#17211)

### Feature improvements

#### Automatic Mixed Precision

 - [AMP] Move topk from FP16_FP32_FUNCS to FP32_FUNCS (#15342)
 - Conversion from FP32 model to Mixed Precision model (#15118)
 - Update fp16 docs: Block.cast is inplace (#15458)
 - FP16 Support for C Predict API (#15245)
 - Add AMP Conversion support for BucketingModule (#15528)

#### Gluon Fit API

 - Fixing build for gluon estimator test, including libtvm in pack libs (#16148)
 - [Estimator] handle composite metrics in estimator (#16676)
 - [Estimator] refactor estimator to allow overriding evaluate/fit of a batch (#16678)
 - [Estimator] refactor estimator and clarify docs (#16694)
 - [Gluon] Improve estimator usability and fix logging logic (#16810) (#16846)
 - Backport Gluon estimator changes to 1.6 (#17048)
 - fix parameter names in the estimator api (#17051) (#17162)


#### MKLDNN

 - Upgrade MKL-DNN submodule to v0.20 release (#15422)
 - Fix quantized concat when inputs are mixed int8 and uint8 (#15693)
 - [MKLDNN]Enhance Quantization APIs and Tutorial (#15448)
 - Add quantization support for GluonCV (#15754)
 - add int8 bn mkldnn implementation and test (#15664)
 - [Quantization]support exclude operators while quantization (#15910)
 - [MKLDNN]Support fullyconnected and element-wise ops fusion (#15950)
 - Disable test coverage for Clang MKLDNN (#15977)
 - update support MKLDNN BN conditions (#15870)
 - [MKLDNN] Fix out of bound access of req vector (#16000)
 - add uint8 bn mkldnn implementation (#16003)
 - Improve quantization flow (#15961)
 - [MKLDNN] fix uint8 batch norm memory misuse (#16034)
 - MKL-DNN RNN checks NDArray version (#16071)
 - Float64 fallback for mkldnn subgraph and rnn op (#15853)
 - Update MKL-DNN dependency (#16073)
 - Integrate MKL-DNN leakyrelu (#16075)
 - [MKLDNN] NDArray reorder in C API and deconv (#16265)
 - Fix mkldnn reshape (#16455)
 - [MKLDNN] Fix uint quantized fc when not fusing with requantize (#16523)
 - [MKLDNN]Fix reorder2default (#16602)
 - Upgrade MKL-DNN dependency to v1.0 (#16555)
 - Revert "[MKLDNN]Fix reorder2default (#16602)" (#16697)
 - [v1.6.x] Backport #16837 into v1.6.x (#16847)
 - Initial checkin (#16856) (#16872)

#### Large tensor support

 - [MXNET-1413] Adding Large Tensor support for sort operators (#15170)
 - Large Index Support for Slice (#15593)
 - Add large tensor support binary arithmetic (#15785)
 - Large tensor support for random ops (#15783)
 - Add Large Tensor Support for Sequence, NN Ops  (#15807)
 - Add power, exponent, log ops large tensor support (#15794)
 - removing unnecessary int64 C apis that were added to support Large Tensors and Vectors (#15944)
 - creating ndarray directly using mxnet ndarray primitives to reduce memory footprint of tests for topk, sort and argsort (#15900)
 - Adding tests to verify support for Large Tensors in additional Ops along with new C_Apis supporting 64bit indexing (#15895)
 - Added tests to verify Large Vector Support for initial set of ops  (#15943)
 - Added more tests for Large Indices (#15960)
 - Add Large tensor vector test cases (#15941)
 - Test large vector mean operator and fix a few bugs (#16079)
 - Reducing memory footprint of one_hot for Large Array Testing (#16136)
 - removing MXNDArrayLoadFromBuffer64 and MXNDArrayLoad64 (#16203)
 - Fix large array tests (#16328)
 - added more tests to verify support for large vector (#16477)
 - added support for large tensors for Dropout operator and tests to verify support for more operators (#16409)
 - adding large tensor support for add_n and tests for more ops (#16476)
 - adding large tensor support for pad operator (#15126)
 - Added large tensor support and test for gather_nd (#16371)
 - Large Vector tests for DGL Ops Part 2 (#16497)
 - Showing proper error message when an attempt is made to create large tensor but MXNet is not built with it (#16570)

#### TensorRT integration

 - enable TensorRT integration with cpp api (#15335)
 - Add unit tests for TensorRT integration and fix some bugs (#15399)

#### Higher order gradient support

 - [MXNET-978] Higher order gradient for sigmoid (#15288)
 - [MXNET-978] Higher Order Gradient Support `reciprocal`, `abs`. (#15413)
 - [MXNET-978] Add higher order gradient support `tan`, `tanh` (#15253)
 - [MXNET-978] Higher Order Gradient Support `arctan`, `arctanh`, `radians`. (#15531)
 - [MXNET-978] Higher Order Gradient Support `sqrt`, `cbrt`. (#15474)
 - [MXNET-978] Higher Order Gradient Support `clip`, `dropout`. (#15746)
 - [MXNET-978] Higher Order Gradient Support `sinh`, `cosh`. (#15412)
 - [MXNET-978] n-th order gradient test support. (#15611)
 - [MXNET-978] Fully connected, higher order grad (#14779)
 - [MXNET-978] Higher Order Gradient Support `arcsinh`, `arccosh`. (#15530)

#### Operator improvements

 - broadcast axis is alias to broadcast axes; doc fix (#15546)
 - Utility to help developers debug operators: Tensor Inspector (#15490)
 - Softmax with length (#15169)
 - in-place reshape ops (#14053)
 - Add missing default axis value to symbol.squeeze op (#15707)
 - Add matrix determinant operator in linalg (#15007)
 - Add fp16 support for topk (#15560)
 - [MXNET-1399] multiclass-mcc metric enhancements (#14874)
 - new raise mode for nd.take and fix backward for wrap mode (#15887)

#### Profiler

 - Fixing duplication in operator profiling (#15240)
 - Custom Operator Profiling Enhancement (#15210)
 - [Opperf] Make module/namespace of the operator parameterized (#15226)
 - Opperf: Support Python<3.6 (#15487)
 - Add transpose_conv, sorting and searching operator benchmarks to Opperf (#15475)
 - Deprecate USE_PROFILER flag (#15595)
 - Update profiler.md (#15477)
 - [Opperf] Add array rearrange operators to opperf (#15606)
 - [OpPerf] PDF Random ops fix (#15661)
 - [Opperf] Add  optimizer update operator benchmarks to opperf (#15522)
 - fix broadcast op param (#15714)
 - [OpPerf] Profiler flag for Python, Cpp  (#15881)
 - [Opperf] Filter out deprecated ops (#15541)
 - [OpPerf] Handle positional arguments (#15761)
 - [OpPerf] Take care of 4d param  (#15736)
 - Add Median,p50,p99 to python profiler (#15953)
 - adding "total" (total time) to profiler aggregate stats sorting criteria (#16055)

#### ONNX import/export

 - Correct ONNX documentation (#15914)
 - [MXNET-895] ONNX import/export: TopK (#13627)

#### Runtime discovery of features

 - Making Features as a singleton for improved caching (#15835)

#### Bug fixes

 - [bug] fix higher grad log  (#15120)
 - Showing proper error when csr array is not 2D in shape. (#15242)
 - add 'asnumpy' dtype option to check_symbolic_backward (#15186)
 - point fix the vector declaration in MultiBoxDetection (#15300)
 - Temporarily Commenting out Flaky Test (#15436)
 - Fix memory leak in NaiveEngine (#15405)
 - fix nightly CI failure (#15452)
 - Small typo fixes in batch_norm-inl.h (#15527)
 - Bypass cuda/cudnn checks if no driver. (#15551)
 - Julia path patch (#15561)
 - Fix AMP Tutorial failures (#15526)
 - Fix warnings in CLang: (#15270)
 - Fix dumps for Constant initializer (#15150)
 - fix normalize mean error bug (#15539)
 - [fix] print `self` in warning. (#15614)
 - [MXNET-1411] solve pylint error issue#14851 (#15113)
 - [Flaky test] Skip test_operator_gpu.test_convolution_independent_gradients (#15631)
 - Fix subgraph with custom_op (#15671)
 - Fix USE_BLAS == openblas check (#15691)
 - update previous flaky naive engine test (#15651)
 - make TransposeShape infer shape form both sides (#15713)
 - Skip Flaky Test (#15722)
 - Revert "Dynamic Library Loading Support" (#15755)
 - Fix flaky test test_global_metric (#15756)
 - Fix PR #15489 (Dynamic Library Loading Support) (#15760)
 - Refactor LibraryInitializer so it's thread safe. Fixes random sporadical concurrency crashes. (#15762)
 - Fix backward_clip num inputs and type of clip params (#15688)
 - fixing problem with existing Singleton Caching (#15868)
 - Allow operators with multiple outputs in get_atomic_symbol (#15740)
 - Fix ConcatType backward type inference (#15829)
 - Add disable attr to subgraph property (#15926)
 - Re-enable flaky test_prelu (#15777)
 - declare explicitly the tblob default assign operator and copy constructor (#15937)
 - Discard needless test cases in `test_convolution_independent_gradients` (#15939)
 - fix naive engine for multi-threaded inference (#15574)
 - Fix get_rows_per_block (#15979)
 - Fix a memory misalignment in topk operator (#15948)
 - Decouple dtype from shape for Random multinomial (#15980)
 - Fix dtype inference in arange_like operator (#15930)
 - Disable laop_6 (#15976)
 - Fix flaky clojure profile test (#16058)
 - fix test_pick test time  is too long (#16066)
 - [fix] Support nullop in `transpose` (#15865)
 - fix flaky test (#16074)
 - fix some test files test time is too long (#16067)
 - Fix gradient tensor mutate in `{adam/ftrl/rmprop/rmspropalex}_update`. (#15768)
 - Fix unary operator ceil/floor/trunc when data type is integer (#14251)
 - Fix failing tests (#16117)
 - Fixes NAG optimizer #15543 (#16053)
 - avoid test relu at the origin due to discontinuous gradient (#16133)
 - Fix remaining errors reported by D2L (#16157)
 - use 1E-4 in groupnorm test(#16169)
 - Sequence last fix (#16156)
 - fixing test for model compatibility checker (#16159)
 - assert_allclose -> rtol=1e-10 (#16198)
 - [MEMORY] retry GPU memory allocation if fragmented (#16194)
 - improve dataloader signals and messages (#16114)
 - Update ndarray.py (#16205)
 - fix flaky test (#16191)
 - Solve #14116, #15143 (#15144)
 - [MXNET-1422] Fix wrong results of min([inf, inf]) and max([-inf,-inf]) (#16226)
 - Fix inconsistent interpolation method values (#16212)
 - set fixed seed for profiler (#16155)
 - Fix MXNDArrayGetData (#16289)
 - fix atol for test_preloaded_multi_sgd (#16356)
 - Fix windows flakiness (#16415)
 - cuDNN non-persistant bidirectional RNN dgrad sync fix (#16391)
 - [BUGFIX] Minor type issues in Squeeze (#16448)
 - Fix Nightly Tests for Binaries (#16451)
 - Fix dtype bug (#16467)
 - Fix flakey pylint CI failures (#16462)
 - Load NDArray only to GPU if GPU is present (#16432)
 - Bug fix for the input of same axes of the swapaxes operator (#16513)
 - Fix learning rate scheduler being unexpectedly overwritten by optimizer's default value (#16487)
 - disable tests (#16536)
 - fix pylint in CI (#16540)
 - image crop gpu (#16464)
 - Build dmlc-core with old thread_local implementation (#16526)
 - fix doc for topk (#16571)
 - RNNOp to call cudaEventCreate lazily (#16584)
 - add encoding to the stub files for potential utf8 char in doc strings (#16580)
 - Surpress subgraph log in CI (#16607)
 - Fix dequantize memory corruption (#16606)
 - Fix for wrong reqs set after switching from training to inference (#16553)
 - Disables test_bulking_operator_gpu due to flakiness (#16611)
 - Imagenet inference to nightly fix (#16599)
 - Move some subgraph verbose to MXNET_SUBGRAPH_VERBOSE=2 (#16622)
 - RNNOp only call cuda/cudnn if GPU ctx is requested (#16632)
 - fix bad encode (#16641)
 - Disable float16 test (#16643)
 - Fix GetMKLDNNData for delay alloc (#16618)
 - Move ops which don't support FP16 dtype to FP32 list (#16668)
 - no such method => modified function args (#16610)
 - fix cuDNN RNN dtype_with_fallback_ bug (#16671)
 - Add check if scipy is imported in sparse.py (#16574)
 - Added launch bounds to the reduce kernels (#16397)
 - fix install dir (#16690)
 - fix binary dependencies in CD and nightly (#16693)
 - Fix SliceChannel Type inference (#16748) (#16797)
 - fix flakiness of test_np_mixed_precision_binary_funcs (#16873)
 - Fix test_gluon.py:test_sync_batchnorm when number of GPUS > 4 (#16835)
 - Omp fork numthreads fix 1.6 (#17000)
 - [BUGFIX] Fix race condition in kvstore.pushpull (#17007) (#17052)
 - Backport #17002, #17068 and #17114 to 1.6 branch (#17137)
 - Backport 3rdparty/openmp fixes (#17193)
 - fix norm sparse fallback (#17149)

### Front end API

 - Expose get_all_registered_operators and get_operator_arguments in the… (#15364)
 - Add magic method `abs` to NDArray and Symbol. (#15680)
 - Dynamic Library Loading Support (#15489)
 - [MXNET-1294] Add KVSTORE PushPull API (#15559)

#### Gluon

 - [Dataset] Add take, filter, sample API to dataset (#16078)
 - Add register_op_hook for gluon (#15839)
 - [Dataset] add shard API (#16175)
 - Add list_ctx to ParameterDict (#16185)
 - [Gluon] Support None argument in HybridBlock (#16280)
 - Aggregated zero grad (#16446)
 - try to fix block (#16465)
 - [Gluon] Don't serialize shared parameters twice (#16582)
 - Initializer.__eq__ (#16680)

#### Symbol

 - Add symbol api for randn and fix shape issue for randn ndarray and symbol api (#15772)
 - Graph Partition API (#15886)

### Language Bindings

#### Python

MXNet community [voted](https://lists.apache.org/thread.html/r3a2db0f22a1680cc56804191446fef2289595798ca19fd17de1ff03e%40%3Cdev.mxnet.apache.org%3E) to no longer support Python 2 in future releases of MXNet. Therefore, MXNet 1.6 release is going to be the last MXNet release to support Python 2.

#### C/C++

 - [C++] Improve inference script to support benchmark on Imagenet (#15164)
 - C Api for simplebind, fix comment for trigoops, add atol to assert (#16585)

#### Clojure

 - Extend Clojure BERT example (#15023)
 - [Clojure] Add fastText example (#15340)
 - make clojure api generator tests less brittle (#15579)

#### Julia

 - add julia env settings (#15523)
 - julia: bump window prebult binary version to v1.5.0 (#15608)
 - julia: remove Travis CI related files (#15616)
 - julia: bump binding version to v1.6.0 (#15607)
 - julia: rename build env var `MXNET_HOME` to `MXNET_ROOT` (#15568)
 - Revert "julia: rename build env var `MXNET_HOME` to `MXNET_ROOT` (#15568)" (#16147)
 - julia: fix `mx.forward` kwargs checking (#16138)
 - julia: implement `context.num_gpus` (#16236)
 - julia: add `AbstractMXError` as parent type (#16235)
 - [MXNET-1430] julia: implement context.gpu_memory_info (#16324)
 - julia/docs: more DRY on page rendering (#16396)

#### Perl

 - [Perl] - simplify aliasing strategy (#15395)
 - [Perl] - ndarray to native array conversion fix (#16635)

#### Scala

 - Add Sparse NDArray support for Scala (#15378)
 - fix the bug on Scala Sparse (#15500)
 - fix heap-use-after-free in scala (#15503)
 - Bump Scala version to 1.6 (#15660)
 - Fix Scala Symbolic API some/Some typo (#15687)
 - Faster Scala NDArray to BufferedImage function (#16219)

### Performance improvements

 - Proper bulking of ops not using FCompute (#15272)
 - improve layernorm CPU performance (#15313)
 - Efficient MXNet sampling in the multinomial distribution (#15311)
 - Revert default return type for indices in argsort() and topk() back to float32 (#15360)
 - Use omp threads for cpu data loader (#15379)
 - Accelerate ROIPooling layer (#14894)
 - Avoid memory copy for dropout inference (#15521)
 - Add omp parallel optimization for _contrib_BilinearReisze2D (#15584)
 - Softmax optimization for GPU (#15545)
 - Speed up group executor (#16069)
 - FullyConnected Bias performance improvement on GPU (#16039)
 - Embedding gradient performance optimization on GPU (#16355)
 - Faster Transpose 2D (#16104)
 - Pseudo 2D transpose kernel (#16229)
 - Faster general take (#16615)

### Examples and tutorials

 - [TUTORIAL] Gluon performance tips and tricks (#15427)
 - Updating profiler tutorial to include new custom operator profiling (#15403)
 - [TUTORIAL] Gluon and Sparse NDArray (#15396)
 - [TUTORIAL] Revise Naming tutorial (#15365)
 - Revise Symbol tutorial (#15343)
 - Two fixes for info_gan.md example Code (#15323)
 - Rebase #13757 to master (#15189)
 - Tensor Inspector Tutorial (#15517)
 - logging (#15106)
 - update profiler tutorial (#15580)
 - [MXNET-1358] Fit api tutorial (#15353)
 - Tutorials nighly fix (#16179)
 - Update add_op_in_backend.md (#16403)
 - typo fix in r doc lstm tutorial (#16546)
 - [MKL-DNN] Add mxnet mkldnn cmake tutorial (#16688)

### Website and documentation

 - [DOC] Clarify that global pooling is going to reset padding (#15269)
 - Update sparse_retain Documentation (#15394)
 - nano instructions (#15117)
 - remove comments from nano instructions (#15433)
 - REAME   MTCNN   Link URL Error in original website (#15020)
 - Update Horovod docs links in README (#15366)
 - fix doc for sort and argsort (#15317)
 - fix comment (#15481)
 - Improve docs for AMP (#15455)
 - [Doc] Add MKL install method apt/yum into tutorial (#15491)
 - Julia docs (#15454)
 - Docs: Fix misprints (#15505)
 - website build for julia: fix path to be static (#15554)
 - some minor typos/clarifications (#15538)
 - refine Nano setup directions (#15524)
 - [Doc] add squeeze to Array change shape (#15549)
 - fix typo (#15648)
 - Fix url (404 error) (#15683)
 - update julia install doc (#15609)
 - [DOC] refine autograd docs (#15109)
 - [DOC] Fix many arguments in the doc: reshape_like, arange_like, shape_array (#15752)
 - Add Gather_nd Scatter_nd to NDArray API category doc (#15689)
 - [Dependency Update] [Doc] move the general prerequisite software to the top (#15896)
 - typo in docs (#16094)
 - [WIP] New Website: New Docs [1/3] (#15884)
 - [DOC] Fix doc for nn.Embedding, nn.Dense and nd.Embedding (#15869)
 - [DOC] Consistent capitalization: mxnet -> MXNet, scala -> Scala (#16041)
 - New Website: Remove Old Content [2/3] (#15885)
 - New Website: New Pipeline [3/3] (#15883)
 - Update KL Divergence formula (#16170)
 - fix broken links (#16255)
 - redirect to the 404 page (#16287)
 - add google-analytics config (#16271)
 - Fixing links for website + Fixing search (#16284)
 - Minor fix in ToTensor documentation. (#16299)
 - adding redirects so that old website API links surfaced from searches (#16342)
 - Fix code block formatting in Why MXNet doc page (#16334)
 - Julia: add API docs back (#16363)
 - Change mailing list url in footer to point to instructions about how to subscribe instead (#16384)
 - Add instructions to report a security vulnerability (#16383)
 - [DOC] fix installation selector wrong history (#16381)
 - Beta build (#16411)
 - [WIP] Improving Python Docs API (#16392)
 - fix autodoc for spurrious toggles (#16452)
 - [Doc] Update the download page with 1.5.1 release (#16442)
 - Fixing broken links (#16500)
 - add binary and docs build command options (#16514)
 - add option to remove indexes (#16525)
 - Correct Google Analytics Tracker (#16490)
 - [Doc] Use mirror link in the download page (#16501)
 - checking broken link fixes work (#16538)
 - detect number of procs during sphinx build (#16512)
 - fixed broken links across multiple files (#16581)
 - fix missing docs due to git add issues (#16496)
 - second round of fixing broken links in multiple files (#16598)
 - Python Docstring Convetion (#16550)
 - [MXNET-1434] Fix a broken link for basic C++ tutorial (#16461)
 - Fix python doc build issue (#16630)
 - fixing broken links in multiple files - round 3 (#16634)

### CI/CD

 - Fix build_ccache_wrappers: (#14631)
 - Remove mhard-float option. This is already deprecated by Google. (#15435)
 - CI: upgrade Julia version from 1.0.3 to 1.0.4 (#15502)
 - Add -R option to ci/build.py to avoid rebuilding containers (#15426)
 - [Dependency Update] Bump up the CI Nvidia docker to CUDA 10.1 (#14986)
 - fixed config.mk and Makefile bugs for installing mkl (#15424)
 - Add -DMXNET_USE_OPENMP to Makefiles so libinfo gets updated accordingly (#15498)
 - [Dependency Update] Dependency update doc (#15045)
 - Remove Scala package test on build (#15915)
 - Refactor for windows CI 'out of heap space' errors (#15922)
 - Fix Nightly Maven GPU (#15989)
 - Windows cmake flags cleanup (#16013)
 - Disable flaky test in test_amp_conversion (#16031)
 - Updates git_init Jenkins utility function to support checking out a particular commit id
 - Adds artifact repository scripts
 - Adds CD pipeline framework
 - Adds static libmxnet release pipeline
 - Updates CD pipeline
 - Adds documentation
 - Updates kvstore functions to use pushd and popd
 - Throws exceptions instead o magic numbers
 - Updates artifact repository cli to use --libtype instead of --static or --dynamic
 - Clarifies ci_utils and cd_utils origin remark
 - Adds clarifying note on why ubuntu 14.04 is being used for compilation
 - Removes MXNET_SHA
 - Removes set_release_job_name
 - Adds license headers
 - Updates artifact repository to expect licenses
 - Moves ci/cd to cd directory
 - Takes downstream job name from environment
 - Updates order of parameters
 - Updates job type parameter to dropdown
 - Adds libmxnet feature extraction code comments
 - Removes ccache setup from static build
 - Disable test coverage of C++ codebase on CI  (#15981)
 - Update readme and project.clj comment (#16084)
 - Enable tvm_op for ci (#15889)
 - Not to search for coverage files when none exist (#16107)
 - Fixes openblas installation for static build
 - Update python dependencies (#16105)
 - CD Fixes (#16127)
 - Adds dynamic libmxnet to CD pipeline (#16163)
 - Fix README Build Status (#16183)
 - subscribe to build and CD changes (#16192)
 - [CD] Add COMMIT_ID param to release job (#16202)
 - Fix lack of dylib support in Makefile when use lapack (#15813)
 - Removes git status update stop gap solution (#16285)
 - add mkl installation temp fix (#16304)
 - add 'Release' cmake flag (#16294)
 - S3 upload artifacts (#16336)
 - Fix nightly scala pipeline (#16362)
 - remove redundant branch name (#16372)
 - Skipping installing nightly test (#16418)
 - Adds PyPI CD Pipeline (#16190)
 - upgrade the pytest version (#16429)
 - Revert "add mkl installation temp fix (#16304)" (#16369)
 - increase docker cache timeout (#16430)
 - Adds pip requirements file to nightly gpu ci image (#16472)
 - [CD] Adds python docker pipeline (#16547)
 - Move imagenet inference to nightly (#16577)
 - Backport #16980 #17031 #17018 #17019 to 1.6 branch (#17213)

### Misc

 - update committer info (#15289)
 - Typo fix in plan_memory relase -> release. (#15299)
 - indent changes (#15321)
 - Had a few PRs merged. Hope to become an official contributor and potentially a commiter. (#15451)
 - cuda/cuDNN lib version checking.  Force cuDNN v7 usage. (#15449)
 - Improve diagnose.py, adding build features info and binary library path. (#15499)
 - update ratcheck for apache-rat 0.13 release (#15417)
 - add myself to interested modules (#15590)
 - 1.5.0 news (#15137)
 - bump up version from 1.5.0 to 1.6.0 on master (#15072)
 - Remove myself from CODEOWNERS (#15617)
 - remove mshadow submodule
 - import mshadow source tree
 - cuDNN support cleanup (#15812)
 - Remove requests_failed_to_import handling
 - Update CODEOWNERS. (#15972)
 - Improve diagnose.py to display environment variables (#15715)
 - Update README.md (#16035)
 - [Dev] update ps-lite dependency (#15936)
 - Typedef cleanup (#15899)
 - add KEY for Tao Lv (#16081)
 - remove 'foo' and other print msg from test (#16088)
 - Revert accidental change to CMakelists (#16040)
 - Update env_var.md (#16145)
 - Update dmlc-core (#16149)
 - adding codeowners (#16165)
 - Factorize CUDA_KERNEL_LOOP used in CUDA kernels (#16197)
 - add code of conduct and conflict resolution (#16343)
 - simple typo error in NEWS.md (#16344)
 - update NEWS.md and README.md (#16385)
 - split issue templates (#16558)
 - Create SECURITY.md (#16573)

## 1.5.1
Apache MXNet (incubating) 1.5.1 is a maintenance release incorporating important bug fixes and important performance improvements. All users of Apache MXNet (incubating) 1.5.0 are advised to upgrade. You can install Apache MXNet (incubating) 1.5.1 at the usual place. Please review these Release Notes to learn the bug fixes.

### Bug-fixes
* add deconv in TRT subgraph (#15666) (#16043)
* Update TRT tutorial with new APIs (#16044)
* Fix _copy_to on MKLDNN backend (#15637) (#15803)
* Benchmark doc fix (#15769) (#16029)
* remove Julia cat image for license issue (#15964) (#16026)
* added check for empty params file and unknown param (not arg/aux) (#15917)
* fix license issues (#15806) (#15860)
* prevent TRT_Logger to be destroyed before TRT engine (#14898) (#15877)
* [MXNET-1086] added sub and mul to ONNX->TensorRT conversion (#15344) (#15875)
* handle fix_gamma in tensorrt subgraph conversion correctly (#15645) (#15874)
* fix LinearRegressionOutput with empty label (#15620) (#15873)
* [v1.5.x] [MKLDNN] Independent gradients requests check with respect to weights… (#15805)
* fix dropout mask output (#15697) (#15804)
* fix fp32 flatten issue (#15351) (#15802)
* Clojure package remove source images (#15828)
* changed constructor args (#15601) (#15827)
* Add MKLDNN 4c layout to fix gluoncv se_resnext101_64x4d (#15692) (#15801)
* Fix the bug of `MXEnginePushAsyncND` and `MXEnginePushSyncND` (#15751) (#15792)

## 1.5.0

### New Features

#### Automatic Mixed Precision(experimental)
Training Deep Learning networks is a very computationally intensive task. Novel model architectures tend to have increasing numbers of layers and parameters, which slow down training. Fortunately, software optimizations and new generations of training hardware make it a feasible task.
However, most of the hardware and software optimization opportunities exist in exploiting lower precision (e.g. FP16) to, for example, utilize Tensor Cores available on new Volta and Turing GPUs. While training in FP16 showed great success in image classification tasks, other more complicated neural networks typically stayed in FP32 due to difficulties in applying the FP16 training guidelines.
That is where AMP (Automatic Mixed Precision) comes into play. It automatically applies the guidelines of FP16 training, using FP16 precision where it provides the most benefit, while conservatively keeping in full FP32 precision operations unsafe to do in FP16. To learn more about AMP, check out this [tutorial](https://github.com/apache/mxnet/blob/master/docs/tutorials/amp/amp_tutorial.md).

#### MKL-DNN Reduced precision inference and RNN API support
Two advanced features, fused computation and reduced-precision kernels, are introduced by MKL-DNN in the recent version. These features can significantly speed up the inference performance on CPU for a broad range of deep learning topologies. MXNet MKL-DNN backend provides optimized implementations for various operators covering a broad range of applications including image classification, object detection, and natural language processing. Refer to the [MKL-DNN operator documentation](https://github.com/apache/mxnet/blob/v1.5.x/docs/tutorials/mkldnn/operator_list.md) for more information.

#### Dynamic Shape(experimental)
MXNet now supports Dynamic Shape in both imperative and symbolic mode. MXNet used to require that operators statically infer the output shapes from the input shapes. However, there exist some operators that don't meet this requirement. Examples are:
* while_loop: its output size depends on the number of iterations in the loop.
* boolean indexing: its output size depends on the value of the input data.
* many operators can be extended to take a shape symbol as input and the shape symbol can determine the output shape of these operators (with this extension, the symbol interface of MXNet can fully support shape).
To support dynamic shape and such operators, we have modified MXNet backend. Now MXNet supports operators with dynamic shape such as [`contrib.while_loop`](https://mxnet.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.while_loop), [`contrib.cond`](https://mxnet.apache.org/api/python/ndarray/contrib.html#mxnet.ndarray.contrib.cond), and [`mxnet.ndarray.contrib.boolean_mask`](https://mxnet.apache.org/api/python/ndarray/contrib.html#contrib)
Note: Currently dynamic shape does not work with Gluon deferred initialization.

#### Large Tensor Support
Currently, MXNet supports maximal tensor size of around 4 billon (2^32). This is due to uint32_t being used as the default data type for tensor size, as well as variable indexing.
This limitation has created many problems when larger tensors are used in the model.
A naive solution to this problem is to replace all uint32_t in the MXNet backend source code to int64_t.
This solution is not viable, however, because many data structures use uint32_t as the data type for its members.
Unnecessarily replacing these variables to int64_t will increase the memory consumption causing another limitation. Second, MXNet has many submodule dependencies.
Updating the variable types in the MXNet repository is not enough. We also need to make sure different libraries, such as MKLDNN, MShadow etc. supports the int64_t integer data type.
Third, many front end APIs assume unsigned 32-bit integer interface. Only updating the interface in C/C++ will cause all the language bindings to fail.
Therefore, we need a systematic approach to enhance MXNet to support large tensors.
Now you can enable large tensor support by changing the following build flag to 1: `USE_INT64_TENSOR_SIZE = 1`. Note this is set to 0 by default.
For more details please refer to the [design document](https://cwiki.apache.org/confluence/display/MXNET/Large+Tensor+Support).

#### Dependency Update
MXNet has added support for CUDA 10, CUDA 10.1, cudnn7.5, NCCL 2.4.2, and numpy 1.16.0.
These updates are available through PyPI packages and build from source, refer to [installation guide](https://mxnet.apache.org/versions/master/install/index.html) for more details.

#### Gluon Fit API(experimental)
Training a model in Gluon requires users to write the training loop. This is useful because of its imperative nature, however repeating the same code across multiple models can become tedious and repetitive with boilerplate code.
The training loop can also be overwhelming to some users new to deep learning. We have introduced an Estimator and Fit API to help facilitate training loop.
Note: this feature is still experimental, for more details, refer to [design document](https://cwiki.apache.org/confluence/display/MXNET/Gluon+Fit+API+-+Tech+Design).

#### New Operators
* split_v2 (#13687)
* Gradient multiplier (contrib) operator (#13632)
* Image normalize operator - GPU support, 3D/4D inputs (#13802)
* Image ToTensor operator - GPU support, 3D/4D inputs (#13837)
* Add Gluon Transformer Crop (#14259)
* GELU (#14449)
* AdamW operator (Fixing Weight Decay Regularization in Adam) (#13728)
* [MXNET-1382] Add the index_array operator (#14638)
* add an operator for computing the likelihood of a Hawkes self-exciting process (#14683)
* Add numpy linspace (#14927)


### Feature Improvements

#### Operators
* make ROIAlign support position-sensitive pooling (#13088)
* Add erfinv operator for calculating inverse error function (#13811)
* Added optional parameters to BilinearResize2D to do relative scaling (#13985)
* MXNET-1295 Adding integer index support to Sequence* family of operators. (#13880)
* Export resize and support batch size (#14014)
* CUDNN dropout (#13896)
* Relaxing type requirements for slice_like op (#14097)
* Relaxing type requirements for reshape_like op (#14325)
* Parallelize CPU version and add GPU version of boolean_mask op (#14090)
* Add NHWC layout support to Pooling (cpu, gpu cuda, gpu cuDNN) (#13749)
* Multi-precision AdamW update op (#14171)
* [op] add back support for scalar type rescale_grad argument for adamw_update/mp_adamw_update (#14221)
* move choose_element_0index to operator (#14273)
* Optimize NMS (#14290)
* Optimize NMS part 2 (#14352)
* add background class in box_nms (#14058)
* Use cudnn for dropout by default (#14278)
* In-place updates for Nadam, Adadelta, Adamax and SGLD (#13960)
* Aggregate SGD (#13346)
* Add proper exception message for negative shape in array creation routines (#14362)
* Support multi-threading for Custom Operator (#14363)
* moveaxis operator now accepts negative indices and sequence of ints as well. (#14321)
* Support SyncBatchNorm5D (#14542)
* Add nd.power and sym.pow (#14606)
* Change RNN OP to stateful (#14476)
* Add imresize and copyMakeBorder to mx.image (#13357)
* add ctx for rand_ndarray and rand_sparse_ndarray (#14966)
* Add cpu implementation for Deformable PSROIPooling (#14886)
* Add warning for fp16 inputs with MXNET_SAFE_ACCUMULATION=0 (#15046)
* Safe LayerNorm (#15002)
* use MXNET_SAFE_ACCUMULATION for softmax accumulator (#15037)
* LayerNorm acceleration on GPU  (#14935)
* Add matrix inversion operator in linalg (#14963)
* implementation for equivalence of tf.moments (#14842)
* Use env var to enforce safe accumulation in ReduceAxesCompute (#14830)
* [MXNet-1211] Factor and "Like" modes in BilinearResize2D operator (#13226)
* added extraction/generation of diagonal and triangonal matrices to linalg (#14501)
* [Mxnet-1397] Support symbolic api for requantize and dequantize (#14749)
* [MXNET-978] Support higher order gradient for `log`. (#14992)
* Add cpu implementation for Deformable Convolution (#14879)

#### MKLDNN
* Feature/mkldnn static (#13628)
* Feature/mkldnn static 2 (#13503)
* support mkl log when dtype is fp32 or fp64 (#13150)
* Add reshape op supported by MKL-DNN (#12980)
* Move the debug output message into MXNET_MKLDNN_DEBUG (#13662)
* Integrate MKLDNN Conv1d and support 3d layout (#13530)
* Making MKL-DNN default on MXNet master (#13681)
* Add mkldnn OP for slice (#13730)
* mkldnn s8 conv API change for master (#13903)
* [MKLDNN] Enable signed int8 support for convolution. (#13697)
* add mkldnn softmax_output (#13699)
* MKLDNN based Quantized FullyConnected Operator and its fusion (#14128)
* Fix entropy for uint8 (#14150)
* Update MKL-DNN to v0.18 release (was: fix the Dense layer issue) (#13668)
* [MKL-DNN] Enable s8 support for inner product and 3d input with flatten=false (#14466)
* Optimize transpose operator with MKL-DNN (#14545)
* [MKLDNN] Remove repeat parts in MKLDNN.md (#14995)
* [MKLDNN] Enable more convolution + activation fusion (#14819)
* Update MKL-DNN submodule to v0.19 (#14783)
* Add mkldnn_version.h to pip package (#14899)
* [MKLDNN] add quantized sum (#14614)
* [MKLDNN]Refactor requantize to speed up execution (#14608)
* [MKLDNN]Add quantized relu (#14604)
* Add MKLDNN headers to pip package (#14339)
* add symbolic link to mkldnn header files in include (#14300)
* disable default MKLDNN for cross compilation (#13893)
* Update MKLDNN_README.md (#13653)
* [Quantization] Support zero-size tensor input for quantization flow (#15031)
* Support 3D input for MKL-DNN softmax operator (#14818)
* Add primitive cache for MKL-DNN sum(elemwise_add operator (#14914)
* Fix reshape to add in-place back (#14903)
* [int8] Add MobileNetV2_1.0 & ResNet18 Quantization (#14823)
* [MKLDNN]Improve quantizeV2 and dequantize latency (#14641)
* added mkldnn dependency for plugin compile target (#14274)
* Support Quantized Fully Connected by INT8 GEMM (#12922)

#### ONNX
* ONNX export: Instance normalization, Shape (#12920)
* ONNX export: Logical operators (#12852)
* ONNX import/export: Size (#13112)
* ONNX export: Add Flatten before Gemm (#13356)
* ONNX import/export: Add missing tests, ONNX export: LogSoftMax (#13654)
* ONNX import: Hardmax (#13717)
* [MXNET-898] ONNX import/export: Sample_multinomial, ONNX export: GlobalLpPool, LpPool (#13500)
* ONNX ops: norm exported and lpnormalization imported (#13806)
* [MXNET-880] ONNX export: Random uniform, Random normal, MaxRoiPool (#13676)
* ONNX export: Add Crop, Deconvolution and fix the default stride of Pooling to 1 (#12399)
* onnx export ops (#13821)
* ONNX export: broadcast_to, tile ops (#13981)
* ONNX export: Support equal length splits (#14121)

#### TensorRT
* [MXNET-1252][1 of 2] Decouple NNVM to ONNX from NNVM to TenosrRT conversion (#13659)
* [MXNET-703] Update to TensorRT 5, ONNX IR 3. Fix inference bugs. (#13310)
* [MXNET-703] Minor refactor of TensorRT code (#13311)
* reformat trt to use subgraph API, add fp16 support (#14040)

#### FP16 Support
* Update mshadow to support batch_dot with fp16. (#13716)
* float32 → float16 cast consistency across implementations (#13857)
* modifying SyncBN doc for FP16 use case (#14041)
* support dot(vector, vector) for fp16 inputs on GPU (#14102)
* softmax for fp16 with fp32 accumulator (#14098)
* [MXNET-1327] Allow RNN Layers to be initialized to fp16 (#14219)
* fp16 safe norm operator (#14616)
* NAG Optimizer with multi-precision support (#14568)

#### Deep Graph Library(DGL) support
* Add graph_compact operator. (#13436)
* Accelerate DGL csr neighbor sampling (#13588)

#### Horovod Integration
* Add extra header file to export for error checking (#13795)
* whitelist symbols for using MXNet error handling externally (#13812)
* Use CPUPinned context in ImageRecordIOParser2 (#13980)
* Add pin_device_id option to Gluon DataLoader (#14136)

#### Dynamic Shape
* [MXNET-1315] Add checks for dynamic-shaped operators in CachedOp (#14018)
* [MXNET-1325] Make InferShapeAttr a standalone pass (#14193)
* [MXNET-1324] Add NaiveRunGraph to imperative utils (#14192)
* [MXNET-1352] Allow dynamic shape in while_loop and if conditionals (#14393)

#### Backend Engine
* Add infer_type_partial (#14214)
* Tidy up storage allocation and deallocation (#14480)
* Add MXEnginePushAsync and MXEnginePushSync C APIs (#14615)
* Enhance subgraph API (#14113)
* Enhance PartitionGraph (#14277)
* Allow clearing gpu cache (#14252)
* Fix warning / static function in header. (#14900)
* Simplify creation of NodeEntry instances and use emplace_back (#14095)
* Add unpooled gpu memory type (#14716)
* [MXNET-1398] Enable zero-copy from numpy to MXNet NDArray (#14733)
* Use DEFAULT macro in C APIs (#14767)
* Avoid unnecessary vector copies in imperative_utils.cc (#14665)
* Support populating errors back to MXNet engine in callback (#13922)
* Restore save/load ndarray to 1.4.1 (#15073)
* Enable serializing/deserializing ndarrays in np_shape semantics (#15090)
* [numpy] Support zero-dim and zero-size tensors in MXNet (#14661)
* Rename np_compat to np_shape (#15063)
* [MXNET-1330] Bring nnvm::Tuple to mxnet::Tuple (#14270)

#### Large Tensor Support
* Large array support for randint (#14242)
* [MXNET-1185] Support large array in several operators (part 1) (#13418)
* [MXNET-1401] adding more operators to test support for Large Tensor (#14944)
* [MXNET-1410]Adding Large Tensor Support for tensor transpose (#15059)

#### Quantization
* Exclude concat layer for gpu quantization (#14060)
* Enhance gpu quantization (#14094)
* Register fake grad to subgraph and quantized operators (#14275)
* Add int8 data loader (#14123)

#### Profiler
* [MXNET-857] Add initial NVTX profiler implementation (#12328)

#### CoreML
* Add more support for mxnet_to_coreml (#14222)


### Front End API

#### Gluon
* Add pixelshuffle layers (#13571)
* [MXNET-766] add dynamic_unroll RNN for HybridBlock (#11948)
* add pos_weight for SigmoidBinaryCrossEntropyLoss (#13612)
* Rewrite dataloader with process pool, improves responsiveness and reliability (#13447)
* Complimentary gluon DataLoader improvements (#13606)
* [Fit-API] Adress PR comments (#14885)
* [Fit API] update estimator (#14849)
* [MXNET-1396][Fit-API] Update default handler logic (#14765)
* [Fit API] improve event handlers (#14685)
* move to gluon contrib (#14635)
* move estimator to contrib (#14633)
* [MXNet-1340][Fit API]Update train stats (#14494)
* [MXNet-1334][Fit API]base class for estimator and eventhandler (#14346)
* [MXNET-1333] Estimator and Fit API (#14629)
* Add support for fast variable-length LSTM (#14208)
* Add the Gluon Implementation of Deformable Convolution (#14810)
* hybridize rnn and add model graph (#13244)

#### Python
* Python BucketingModule bind() with grad_req = 'add' (#13984)
* Refine runtime feature discovery python API and add documentation to ... (#14130)
* Runtime feature detection (#13549)
* Add dtype visualization to plot_network (#14066)
* [MXNET-1359] Adds a multiclass-MCC metric derived from Pearson (#14461)
* support long for mx.random.seed (#14314)
* Optimization of metric evaluation (#13471)
* [MXNET-1403] Disable numpy's writability of NDArray once it is zero-copied to MXNet (#14948)
* Refactor ImageRecordIter (#14824)


### Language Bindings

#### Scala
* [MXNET-1260] Float64 DType computation support in Scala/Java (#13678)
* [MXNET-1000] get Ndarray real value and form it from a NDArray (#12690)
* Now passing DType of Label downstream to Label's DataDesc object (#14038)
* Scala interpreter instructions (#14169)
* Add default parameters for Scala NDArray.arange (#13816)
* [MXNET-1287] Up scala comp (#14667)
* [MXNET-1385] Improved Scala Init and Macros warning messages (#14656)
* Remove all usages of makefile for scala (#14013)
* Update scala-package gitignore configuration. (#13962)
* [MXNET-1177]Adding Scala Demo to be run as a part of Nightly CI (#13823)
* [MXNET-1287] Miscellaneous Scala warning fixes (#14658)
* Fix jar path and add missing ones for spark jobs (#14020)
* [MXNET-1155] Add scala packageTest utility (#13046)
* [MXNET-1195] Cleanup Scala README file (#13582)
* Add scalaclean to make clean (#14322)
* Add maven wraper to scala project. (#13702)
* Add new Maven build for Scala package (#13819)
* [MXNET-1287] Feat dep (#14668)
* add Apache header on all XML (#14138)
* update the version name (#14076)
* change to compile time (#13835)
* [MXNET-918] Random module (#13039)
* Avoid secondary deployment of package to local (#14647)

#### Java
* [MXNET-1180] Java Image API (#13807)
* [MXNET-1285] Draw bounding box with Scala/Java Image API (#14474)
* Add BERT QA Scala/Java example (#14592)
* [MXNET-1232] fix demo and add Eclipse support (#13979)
* [MXNET-1331] Removal of non-MXNET classes from JAR (#14303)
* Java install info update (#13912)
* [MXNET-1226] add Docs update for MXNet Java (#14395)
* [MXNET-1383] Java new use of ParamObject (#14645)
* MXNET-1302 Exclude commons-codec and commons-io from assembled JAR (#14000)

#### C++
* print error message for mxnet::cpp::Operator::Invoke when failed (#14318)
* build docs with CPP package (#13983)
* Update inception_inference.cpp (#14674)
* Optimize C++ API (#13496)

#### Clojure
* [Clojure] - Add Spec Validations to the Optimizer namespace (#13499)
* [Clojure] Add Spec Validations for the Random namespace (#13523)
* [Clojure] Correct the versions in the README so they correspond to the latest maven.org release ([#13507)
* Port of scala infer package to clojure (#13595)
* Clojure example for fixed label-width captcha recognition (#13769)
* Update project.clj file to use the snapshots repo to be able to pull (#13935)
* [Clojure] Add resource scope to clojure package (#13993)
* [clojure-package] improve docstrings in image.clj (#14307)
* [Clojure] Helper function for n-dim vector to ndarray (#14305)
* [clojure]: add comp-metric based on CompositeEvalMetric (#14553)
* [Clojure] enhance draw bounding box (#14567)
* [Clojure] Add methods based on NDArrayAPI/SymbolAPI (#14195)
* [Clojure] Clojure BERT QA example (#14691)
* [clojure-package][wip] add ->nd-vec function in ndarray.clj (#14308)
* [Clojure] Correct the versions in the README so they correspond to the latest maven.org release (#13507)
* Update version to v1.5.0 including clojure package (#13566)
* [clojure][generator] ndarray/symbol api random merged (#14800)
* upgrade codox to work with lein 2.9.0 (#14133)
* [clojure] fix: image test does not rely on s3 to run (#15122)

#### Julia
* Julia v0.7/1.0 support and drop v0.6 support (#12845)
* Julia: split ndarray.jl into several snippets (#14001)
* Julia: split symbolic-node.jl into several snippets (#14024)
* Julia: rename mx.clip to clamp for NDArray (#14027)
* Julia: add binding for runtime feature detection (#13992)

#### Perl:
* Two more gluon loss classes. (#14194)

#### R
* add NAG optimizer to r api (#14023)
* R-Package Makefile (#14068)


### Performance Improvements

* Less cudaGet/SetDevice calls in Gluon execution (#13764)
* Improve bulking in Gluon (#13890)
* Increase perfomance of BulkAppend and BulkFlush (#14067)
* Performance improvement in ToTensor GPU Kernel (#14099)
* Performance improvement in Normalize GPU Kernel (#14139)
* Bulked op segments to allow Variable nodes (#14200)
* Performance improving for MKL-DNN Quantized FullyConnected (#14528)
* speedup SequenceMask on GPU (#14445)
* Dual stream cudnn Convolution backward() with MXNET_GPU_WORKER_NSTREAMS=2. (#14006)
* Speedup `_contrib_index_copy` (#14359)
* use mkl sparse matrix to improve performance (#14492)
* Re-enable static cached_op optimization (#14931)
* Speed up SequenceReverse (#14627)
* Improve FC perf when no_bias=False (#15033)
* Improve cached_op performance for static mode (#14785)


### Example and Tutorials

* [MXNET-949] Module API to Gluon API tutorial (#12542)
* Support SSD f32/int8 evaluation on COCO dataset (#14646)
* [MXNET-1209] Tutorial transpose reshape  (#13208)
* [Clojure] Add Fine Tuning Sentence Pair Classification BERT Example (#14769)
* example/ssd/evaluate/eval_metric.py (#14561)
* Add examples of running MXNet with Horovod (#14286)
* Added link to landing page for Java examples (#14481)
* Update lip reading example (#13647)
* [MXNET-1121] Example to demonstrate the inference workflow using RNN (#13680)
* [MXNET-1301] Remove the unnecessary WaitAll statements from inception_inference example (#13972)
* Modifying clojure CNN text classification example (#13865)
* [MXNET-1210 ] Gluon Audio - Example (#13325)
* add examples and fix the dependency problem (#13620)
* add quantization example to readme (#14186)
* Add an inference script providing both accuracy and benchmark result for original wide_n_deep example (#13895)
* Update autoencoder example (#12933)
*  #13813 examples with opencv4/origami (#13813)
* [MXNET-1083] Add the example to demonstrate the inference workflow using C++ API (#13294)
* Add tutorial on how to use build from source jar (#14197)
* Gluon end to end tutorial (#13411)
* Update MXNetTutorialTemplate.ipynb (#13568)
* Simplifications and some fun stuff for the MNIST Gluon tutorial (#13094)
* Clarify dependency on OpenCV in CNN Visualization tutorial. (#13495)
* Update row_sparse tutorial (#13414)
* add clojure tutorials to index (#14814)
* Update lstm_crf.py (#14865)


### Website

* Version switching user experience improvements (#13921)
* fix toctree Sphinx errors (#13489)
* fix link (#15036)
* fix website build (#14148)
* Fixed mailing list addresses (#13766)
* website publish updates (#14015)
* use relative links; update links (#13741)
* update social media section (#13705)
* [MXNET] Updated http://data.dmlc.ml/ links to http://data.mxnet.io/ (#15065)

### Documentation
* [MXNET-1402] MXNet docs change for 1.4.1 release (#14949)
* Add API documentation for upsampling operator with examples (#14919)
* Make docblocks for Gluon BatchNorm and SyncBatchNorm consistent with the code (#14840)
* [DOC] Update ubuntu install instructions from source (#14534)
* [Clojure] Better api docstrings by replacing newlines (#14752)
* Fix documentation for bilinear upsampling and add unit test (#14035)
* Updated docs for R-package installation (#14269)
* [docstring] improve docstring and indentation in `module.clj` (#14705)
* The folder python-howto was removed in an earlier commit. The reference to that folder was not removed. Making a PR to remove the reference to this folder to keep documents consistent (#14573)
* Updated documentation about nightly tests (#14493)
* [Doc] Start the tutorials for MKL-DNN backend (#14202)
* [DOC] fix sym.arange doc (#14237)
* fix render issue in NDArray linalg docs (#14258)
* [clojure-package] fix docstrings in `normal.clj` (#14295)
* [DOC] Refine documentation of runtime feature detection (#14238)
* [MXNET-1178] updating scala docs (#14070)
* Fix website scala doc (#14065)
*  Return value docs for nd.random.* and sym.random.* (#13994)
* Fixing the doc for symbolic version of rand_zipfian (#13978)
* fix doc of take operator (#13947)
* beta doc fixes (#13860)
* [MXNET-1255] update hybridize documentation (#13597)
* Update Adam optimizer documentation (#13754)
* local docs build feature (#13682)
* gluon docfix (#13631)
* Added javadocs and improved example instructions (#13711)
* [MXNET-1164] Generate the document for cpp-package using Doxygen (#12977)
* Fix warning in waitall doc (#13618)
* Updated docs for randint operator (#13541)
* Update java setup docs for 1.4.0 (#13536)
* clarify ops faq regarding docs strings (#13492)
* [MXNET-1158] JVM Memory Management Documentation (#13105)
* Fixing a 404 in the ubuntu setup doc (#13542)
* Fix READMEs for examples (#14179)
* [Doc] Add MKL-DNN operator list (#14891)
* Fixed some typos in AvgPooling Docs (#14324)
* doc fix (#13465)
* Change Straight Dope to Dive into Deep Learning (#14465)
* [DEV] update code owner (#14862)
* Add notes about debug with libstdc++ symbols (#13533)
* Mention additional language bindings and add links (#14798)
* add contributors from intel (#14455)
* what's new - add 1.4.0 release (#14435)
* added note about cuda9.2 requirement (#14140)
* Remove unnecessary "also" in README.md (#14543)
* Updated news.md with the latest mkldnn submodule version (#14298)
* add new cloud providers to install page (#14039)
* Update NOTICE (#14043)
* Update README.md (#13973)
* Update profiler doc (#13901)
* Add CODEOWNERS for Julia package (#13872)
* update code owner (#13737)
* Update git clone location to apache github (#13706)
* NEWS.md backport from v1.4.x to master (#13693)
* Update CODEOWNERS, add Pedro Larroy. (#13579)
* [MXNET-1225] Always use config.mk in make install instructions (#13364)
* Docs & website sphinx errors squished 🌦  (#13488)
* add Qing's Key to master (#14180)
* add KEY for zachgk (#14965)
* corrected a spellign (#14247)
* 1.4 release (#14297)


### Build and Test

* Fix scala doc build break for v1.3.1 (#13820)
* Adds additional CUDA build environments (#14909)
* Pins version of scikit-learn for python2 due to drop in support (#14928)
* upgrade the libpng to 1.6.35 (#14620)
* Updates to cudnn package installation (#14923)
* Improve order of execution of install scripts. (#14867)
* Installs qemu pip requirements from qemu requirements file (#14355)
* update raspberry pi install instructions (#14172)
* update the scala installation tutorial on intellij (#14033)
* Removes unneeded nvidia driver ppa installation (#13814)
* script for installing gpu libraries and build tools (#13646)
* Set install path for libmxnet.so dynamic lib on Mac OS (#13629)
* compatibility with opencv4 (#14313)
* Flaky test #14189 (#14190)
* Enforce determinism for backwards compatibility checker (#14463)
* Change CUB submodule to track Nvidia CUB project. (#13322)
* Updates gpu tests to use CUDNN_VERSION supplied by the environment but default to 7.0.3 if not set (#14595)
* upgrade the version to 2.0.2 (#14621)
* [Dependency Update] Upgrade the libtiff to 4.0.10 (#14623)
* [Dependency Update] Upgrade cuDNN & NCCL (#14884)
* [Dependency Update] Upgrade openssl to 1.1.1b (#14837)
* [Dependency Update] Upgrade CI to use latest cuDNN (#14950)
* GPU RNN to use TempSpace resource for workspace. (#15056)
* Add vim-nox to ci/docker/install/ubuntu_core.sh (#14632)
* Fix dockerized GPU builds in dev_menu (#14603)
* [MXNET-1093] Add python3 Docker images for each MXNet release (#12791)
* increased docker shared memory (#14119)
* Fix permissions of ci/docker/install/ubuntu_publish.sh (#13840)
* Dockerfiles for Publish Testing (#13707)
* Fix test randint (#14990)
* Silence excessive mkldnn logging output on tests. (#14947)
* Fix test memory with ResourceScope (#14666)
* Sync Horovod distributed training examples with latest changes (#14748)
* use mx.context.num_gpus instead of mx.test_utils.list_gpus in MF recommender example (#14926)
* [MXNET-1400] adding tests cases to verify large tensor support for depth_to_space and space_to_depth (#14797)
* rewrite test_custom_op_exc (#14878)
* [Clojure] Remove unneeded test files (#14813)
* Use correct stash name when running nightly tests (#14809)
* julia/ndarray: fix flaky test cases for `clamp` (#14776)
* Updates tolerances for test_layer_bidirectional (#14682)
* Adds context parameter to check_rnn_layer_forward calls in test_lstmp (#14529)
* reenable the test (#14483)
* temporarily disable integ tests with a dependency on origami repo (#14448)
* Bypass ThreadedEngine in test_operator_gpu.py:test_convolution_multiple_streams. (#14338)
* Updated the MLP test to accept the number of epochs. Reduced the epochs in ci_test.sh to shorten the CI build time (#14149)
* follow up on fix nightly test (#14134)
* Julia: enable integration test (#14025)
* fix test_depthwise_convoltuion for occasional CI failures (#14016)
* fix test_stn (#14063)
* Add a test for SGLD optimizer with comparisons for set noise seeds. (#13762)
* Code modification for  testcases of various network models in directory example (#12498)
* Remove MXNET_STORAGE_FALLBACK_LOG_VERBOSE from test_autograd.py (#13830)
* [MXNET-1263] Unit Tests for Java Predictor and Object Detector APIs (#13794)
* ONNX test code cleanup (#13553)
*  #13385 [Clojure] - Turn examples into integration tests (#13554)
* add cpp example inception to nightly test (#13534)
* Fix flaky test test_random:test_randint_generator (#13498)
* Adding test for softmaxoutput (#13116)
* [MXNET-1235] Add a test for AdaMax optimizer (#13467)
* [MXNET-545] Fix broken cython build (#10951)
* Update mkldnn window build instructions in MKLDNN_README.md (#14952)
* Added USE_SIGNAL_HANDLER to other Linux builds which didn't had it (#14122)
* Static build for Python (#13916)
* Julia: add windows-cpu build (#13937)
* Static build instruction for MXNet in general (#13914)
* Jenkins nightly maven with static build script and gpu (#13767)
* Re-organize Scala maven build (#13626)
* disable error checking when building old versions (#13725)
* scripts for building libmxnet binary and wheel (#13648)
* Improve dev_menu usability, local build and virtualenv (#13529)
* Scripts for building dependency libraries of MXNet (#13282)
* [MXNET-1224]: improve scala maven jni build and packing. (#13493)
* fix compile error in debug mode (#13873)
* add ccache to docs build (#13832)
* Decreases test sensitivity (#15014)
* bump up atol for test_bilinear_resize_op (#15011)
* Add STL checks via -D_GLIBCXX_ASSERTIONS in debug mode (#14896)
* clean up duplicate cudnn installation (#14996)
* fix custom op fork test (#14753)
* fix pi instructions (#14746)
* Reenable TensorRT step (#14654)
* Fixes for CI downloads (#14504)
* Fixed tutorial warnings (#14472)
* Fixes static build script for cub directory rename (#14578)
* add a compiler flag to use int64 as tensor size (#14570)
* Upgrade Pylint version to 2.3.1 (#14807)
* Fixes installation nightly test by filtering out the git commands (#14144)
* fix nightly test on tutorials (#14036)
* Fix MXNet R package build (#13952)
* re-enable test after issue fixed https://github.com/apache/mxnet/issues/10973 (#14032)
* Add back R tests and fix typo around R and perl tests (#13940)
* Fix document build (#13927)
* Temporarily disables windows pipeline to unblock PRs (#14261)
* Fix USE_ONEDNN check in Makefile (#13775)
* Fix spelling in threaded_engine_test (#14709)
* Fix cmake options parsing in dev_menu (#13458)
* Add Local test stage and option to jump directly to menu item from commandline (#13809)
* Add CPU test coverage and refine cmake builds (#13338)
* ONNX test code cleanup - part 2 (#13738)
* Rearrange tests written only for update_on_kvstore = True (#13514)
* add batch norm test (#13625)
* Adadelta optimizer test (#13443)
* Skip flaky test https://github.com/apache/mxnet/issues/13446 (#13480)
* Comment out test_unix_python3_tensorrt_gpu step (#14642)
* Enable bulking test on windows (#14392)
* rewrote the concat test to avoid flaky failures (#14049)
* #13624 clojure nightly tests (#13624)
* Temporarily disable website testing (#13887)
* adding tolerance to flaky test (#13850)
* Add publish test of PyPi cu100mkl (#14637)
* CMake: Enable installation of cpp-package headers (#13339)
* Use USE_SIGNAL_HANDLER by default set to ON in CMakeLists.txt (#14599)
* Improve CMake handling of sse2 and sse3 (#14757)
* Update base CUDA image for CI to v10.0 cuDNN 7.3.1 (#14513)
* Updates build_lib.sh to copy the cub library license (#14347)
* Add license check to dev_menu, docs build with docker (#14166)
* Print reproduction command on CI failure (#14815)
* change mxnet_option behavior (#14743)
* [DEP] upgrade dmlc-core (#14510)
* Use ubuntu_rat container for rat check (#14678)
* Added repeats for github status updates (#14530)
* add filter to warnings (#14532)
* CI Changes for Codified Windows AMIs (#14336)
* Refactors USE_NVRTC setting to ENABLE_CUDA_RTC in pip make config files (#14250)
* pypi package description. manifest/setup.py update (#14255)
* make rat-excludes compliant with apache release policy (#14142)
* Add libhdf5-dev to ubuntu_core.sh (#14079)
* Added logging to GitHub commit status publishing (#13615)
* [CI] Prevent timeouts when rebuilding containers with docker. (#13818)
* [MXNET-862] Basic maven jenkins pipeline (#13450)
* Scope requests so it's not needed for dev_menu (#13771)
* Add timeout/retry logic to docker cache download (#13573)
* turn on Sphinx warnings as errors (#13544)
* [MXNET-1251] Basic configuration to do static-linking (#13621)
* Improve CCache handling (#13456)
* build config for maven and pip (#13556)
* Add Intel MKL blas to Jenkins (#13607)
* Add workspace cleaning after job finished (#13490)
* Add a retry to qemu_provision (#13551)
* Deprecate Jenkinsfile (#13474)
* [MXNET-1408] Adding test to verify Large Tensor Support for ravel and unravel (#15048)
* move amp test and change op support to warning (#15085)
* Fixes call to build ubuntu gpu in nightly tests (#14964)
* rat check make target (#15127)
* add epsilon for tolerance level (#15098)
* Change mx.test_utils.list_gpus to mx.context.num_gpus where possible (#14946)
* bump up cudnn to 7.5.1 & nccl 2.4.2 (#14988)
* Disables TensorRT build step (#14958)
* disable flaky integration test (#14151)
* Disables large tensor size cpu test step (#14982)
* Disable Flaky Test test_poisson_generator (#14540)
* Disabled flaky test test_negative_binomial_generator (#13784)
* Disabled flaky test test_gluon_data.test_recordimage_dataset_with_data_loader_multiworker (#13527)


### Bug-fixes

* Improve dev_menu virtualenv handling (#14788)
* Fallback to dense version for grad(reshape), grad(expand_dims) (#13599)
* Fix the bug of BidirectionalCell (#13575)
* set _scale in Trainer using optimizer rescale_grad (#14593)
* [MXNET-1379] update reshape operator (#14600)
* Add repr for SymbolBlock (#14423)
* Cudnn conv dgrad algo filtering (#14310)
* Fix memory leak for size-zero ndarray (#14365)
* Fixes the test_sgld (#14473)
* Revert "Fix memory leak for size-zero ndarray (#14365)" (#14477)
* fix custom operation in fork (#14451)
* Fixes test_operator_gpu.test_multinomial_generator (#14475)
* support leading dimension of -1 in ravel/unravel (#14356)
* begin=end not a valid input (#14403)
* Fix NaN value comparisons in relu, max and min ops (#14262)
* fix engine crash in shutdown phase (#14382)
* fix OOM error during resource allocation (#14444)
* Fix relative difference scala (#14417)
* Correct update count with Gluon trainer and update_on_kvstore=False (#14377)
* Fix crashes on visualization (#14425)
* Reorder module import orders for dist-kvstore (#13742)
* Fixes for trainer with update_on_kvstore=False (#13721)
* Fix errors in docstrings for subgraph op; use code directive (#13463)
* Add resiliency to onnx export code (#13426)
* update github location for sampled_block.py (#13508)
* Revert "Manually track num_max_thread (#12380)" (#13501)
* Revert "Feature/mkldnn static 2 (#13503)" (#13540)
* [MXNET-1110] Add header files required by horovod (#13062)
* [MXAPPS-1020] Clean up some Sphinx warnings. (#13539)
* [MXNET-1249] Fix Object Detector Performance with GPU (#13522)
* [MXNET-769] Use MXNET_HOME in a tempdir in windows to prevent access denied due t… (#13531)
* Chi_square_check for discrete distribution fix (#13543)
* Fix use-before-assignment in convert_dot (#13511)
* fix the situation where idx didn't align with rec (#13550)
* fix link for gluon model zoo (#13583)
* Fix exception handling api doc (#13519)
* [MXNET-1253] fix control_flow_op (#13555)
* fix the Float not showing correctly problem (#13617)
* fix quantize pass error when the quantization supported Op are excluded in the model (#13596)
* Fix for import mxnet taking long time if multiple process launched (#13602)
* Revert "Feature/mkldnn static (#13628)" (#13638)
* updated reference to Apache MXNet (#13645)
* Fix incorrect delete in MXExecutorReshape exception handling (#13376)
* add build fix for Scala/Java build (#13655)
* remove omp which can cause ssd accuracy variance (#13622)
* Fix Jetson compilation (#13532)
* Revert "Fix Jetson compilation" (#13665)
* Fix Jetson compilation (#13666)
* Revert "Revert "[MXNET-43] Fix Jetson compilation" (#13665)" (#13672)
* fix unpicklable transform_first on windows (#13686)
* Fix NDArray ToDLPack Bug (#13698)
* Fix the quantization script to support Python2 (#13700)
* Update basic_layers.py (#13732)
* [MXNET-1231] Allow not using Some in the Scala operators (#13619)
* [MXNET-244] Work around likely compiler bug on nested inlines and temporary acces… (#13535)
* Use curl to download sample data instead of wget. (#13761)
* fix bipartite match memory corruption (#13727)
* remove attributes clear on TRT nodes for GetOptimizedSymbol (#13703)
* fix redirection issues; set default version to master (#13796)
* fix for params with no dims in onnx (#13413)
* Remove semicolon in libmxnet.sym file (#13822)
* remove useless code (#13777)
* Fixing a symlink issue with R install (#13708)
* fix minor indentation (#13827)
* Fix Tree Reduction on new instance type p3dn.24xlarge (#13852)
* [Clojure] package infer tweaks (#13864)
* Fix cpp examples build on Mac. (#13826)
* Fix launch bounds in spatial transformer (#13188)
* Update example scripts classpath. (#13849)
* fix ssd quantization script error (#13843)
* Avoid adding SegfaultLogger if process already has sig handler. (#13842)
* fix the fetching GPU problem (#13889)
* Fix SN-GAN example doc (#13877)
* update Spectral Normalization Code (#13868)
* Fixed java benchmark failing error by fixing the classpath (#13891)
* Fix the order of error term's operands (#13745)
* fix bug in nag optimizer (#13683)
* Fix BatchNorm converter for CoreML when fix_gamma=True (#13557)
* Fix for test always returning true (#13911)
* Add error checking for cpp examples. (#13828)
* julia: fix `argmax` for NDArray (#13871)
* test_ImageRecordIter_seed_augmentation flaky test fix (#12485)
* Julia: fix filename quoting in docstring (#13894)
* Flaky maven binary download (#13974)
* [MXNET-1293] Adding Iterables instead of List to method signature for infer APIs in Java (#13977)
* Sample python bilinear initializer at integral points in y-direction (#12983)
* Fix inconsistent handling for FResourceRequestEx for imperative and symbolic executor (#14007)
* [MXNET-1258] fix unittest for ROIAlign Operator (#13609)
* Fix performance regression in normalize operator (#14055)
* Remove inplace support for ToTensor operator (#14083)
* Addresses comments in runtime feature discovery API (#13964)
* The latest version of leiningen has a dependency problem with codox (#14132)
* Fix quote on LBSGD docs (#13975)
* Fixes spelling (#14168)
* Fix broken amalgamation (#12792)
* Fix nd.pick large array issue (#14082)
* Fix req=null in SliceLikeBackward (#14209)
* onnx broadcast ops fixes (#13604)
* fix update params (#14218)
* MXNet Java bug fixes and experience improvement (#14213)
* reverting broadcasting fixes (#14299)
* fix memory-related issues to enable ASAN tests (#14223)
* FIX: flaky test exponential generator (#14287)
* fix SoftmaxOutput resource bug (#14302)
* Fix shape inference pass (#14153)
* Limit workspace for cudnnGet results (#14326)
* #14199: catch subprocess.CalledProcessError in get_gpus() (#14212)
* Fixes #14181, validate model output shape for ObjectDetector. (#14215)
* Optimizer MXKVStoreUpdater bug fix in serializeState method (#14337)
* Add proper exception message for negative shape in array creation routines (#14362)
* Fix NaN value comparisons in relu, max and min ops (#14262)
* fix engine crash in shutdown phase (#14382)
* Flaky test #14189 (#14190)
* Correct update count with Gluon trainer and update_on_kvstore=False (#14377)
* Fix relative difference scala (#14417)
* fix OOM error during resource allocation (#14444)
* Fix crashes on visualization (#14425)
* begin=end not a valid input (#14403)
* Fix memory leak for size-zero ndarray (#14365)
* Fixes the test_sgld (#14473)
* Revert "Fix memory leak for size-zero ndarray (#14365)" (#14477)
* fix custom operation in fork (#14451)
* Fixes test_operator_gpu.test_multinomial_generator (#14475)
* Fix script retrieval (#14519)
* Memory fixes. Resolves #10867, and resolves #14080 (#14372)
* Chouffe/clojure fix tests (#14531)
* [clojure][image] add draw-bounding-box interop (#14533)
* fix tests (#14565)
* Do not touch GPU 0 during ReleaseAll (#14550)
* [MXNET-1357] Fix the cpp-examples to add exception handling (#14441)
* fix build cpp examples option (#14562)
* Fix flaky test poisson generator & test_negative_binomial_generator (#14571)
* Fixing unintentional variable overloading (#14438)
* fix quantize graph pass (#14605)
* replace std::random_shuffle to std::shuffle (#14523)
* Add exception handling support for waitall (#14397)
* split_and_load can now handle num_ctx > num_data. Issue #13909 (#14607)
* Fix aspect ratio sampling for RandomResizedCrop (#14585)
* [MXNET-400] support string type for kvstore key in cpp-package (#10792)
* Fix warning on macro expansion using defined. (#14598)
* Fix scaladoc scalastyle violations in Infer package (#14671)
* Fix profiler check (#14677)
* Tweak the copy for the cudnn autotuning warning. (#14680)
* Properly handling custom op exception by modify engine (#14693)
* Disable USE_GPERFTOOLS (#14711)
* Reference engine from chunk via weak pointer (#14591)
* [C++] fix type inconsistent issue when loading quantized parameters (#15038)
* Fix crash in random.shuffle operator (#15041)
* [MXNET-1406] [BUG] Fix DLManagedTensor deleter (#15016)
* Fixes lint issue in AMP (#15015)
* Fixed issue where the estimator was printing beyond the dataset size ... (#14464)
* Fixes cuDNN version for CUDA 9.0 build environment (#15001)
* Fix the incorrect MKLDNN/MKL logic in cmake  (#14877)
* Fixed and re-enables TensorRT steps (#14960)
* Fix the return type of sparse.clip operator (#14856)
* Fix sample_multinomial number of outputs bug (#14873)
* [MXNET-13578] Fix cmake installation failed (#14692)
* Fix iterator over symbol when multiple children have the same name (#14597)
* Fixes for wine detection tutorial (#13886)
* Scala/Java Predict API fix #14756 (#14804)
* Fix GELU backward possible NaN (#14782)
* fix shape index bug (#14518)
* [BUGFIX] fix ELU function will appear nan when calculating the gradient (#14673)
* Change size_t to int within for loop to fix windows build error (#14740)
* [contrib][op] fix MultiBoxPrior confusing results if first ratio is not 1.0 (#13763)
* Fix scalastyle (#14669)
* fix Makefile (#14424)
* [v1.4.x] Update MKL-DNN to fix the OSX build issue (#14141) (#14182)
* add preprocessed data and pretrained model info; minor format/spelling fixes (#14170)
* Fixes libjpeg-turbo dependency under Ubuntu 16.04 (#14127)
* Fix website error pages (#13963)
* fix Makefile for rpkg (#13590)
* fix c complier to clang (#13778)
* Fix #13521 (#13537)
* [MXNET-1234] Fix shape inference problems in Activation backward (#13409)
* Revert the change broadcast_to param shape (#14998)
* Fix infer shape partial after unknown shape changed to -1 (#14869)
* fix add_n bug: when input mem overlap with output mem, results is wrong (#14889)
* [Bugfix] Fix layer norm for large input shape (#14870)
* Fix Clojure BERT example's context argument (#14843)
* fix min max on zero-sized ndarray (#14745)
* fix acc_type_switch macro with extra tests (#14773)
* fix bug in profiler tutorial when using cpu (#13695)
* [MXNET-1291] solve pylint errors in examples with issue no.12205 (#13815)
* data preparation file moved in example (#14781)
* [MXNET-1291] solve pylint errors in examples with issue no.12205 (#13848)
*  Prevent crashes for opencv exception and std::exception (#14433)
* Set idx2name for Optimizer object (#14703)
* Revert "Bumped minor version from 1.4.0 to 1.5.0 on master, updated License file" (#13558)
* [BUGFIX] fix unknown parameter shapes when np_shape is turned on. (#15097)
* Add gluonCV to fix AMP Tutorial (#15039)
* fix the if condition for LayerNorm (#15094)
* [MKLDNN]Fix mkldnn deconvolution forward with bias (#15088)
* NER example: fix divisions by zero (#15068)
* remove warning in tutorial: (#15135)
* [MXNET-1291] solve pylint errors in examples with issue no.12205 (#13938)
* Revert "Improve cached_op performance for static mode (#14785)" (#14868)
* Fix mkldnn backend when using naive engine (#15089)
* fix gluon rnn cell single step unroll (#15081)
* Revert "Improve FC perf when no_bias=False (#15033)" (#15099)


### License

* Updates python setup.py for recent license changes (#14778)
* [MXNET-1377] Add static-dependencies licenses (#14726)
* add license (#13793)
* License update  (#13565)
* Bumped minor version from 1.4.0 to 1.5.0 on master, updated License file (#13478)
* License Googletest and Appendix (#14687)
* Add copyrights for third party licenses to license file (#13851)
* Improve license_header tool by only traversing files under revision c… (#13803)
* Update LICENSE File with subcomponents (#13808)

### Depreciations

* Julia: deprecate `mx.empty`, replace it with `UndefInitializer` (#13934)
 * Deprecate NDArrayCollector and instead use ResourceScope (#14780)

### Known Issues
* Amalgamation compile problems(#14808)
* Dynamic Shape does not support reverse shape inference and deferred initialization. (#14983)
* Disables flaky test_random_size_crop (#15019)
* Disables flaky test_l2_normalization (#15006)
* Disables flaky TestStochasticTiming_2D test (#14412)
* Disables flaky test_operator.test_sgld test (#14410)
* Disables test_bulking due to flakyness (#14971)
* Disabled flaky test (#13758)
* Disables flaky test_droupout (#15003)
* Disables flaky test_operator_gpu.test_activation (#14969)


## 1.4.1

Apache MXNet (incubating) 1.4.1 is a maintenance release incorporating important bug fixes and important performance improvements. All users of Apache MXNet (incubating) 1.4.0 are advised to upgrade. You can install Apache MXNet (incubating) 1.4.1 at the usual place. Please review these Release Notes to learn the bug fixes.

### Bug-fixes
* Java bug-fix cherry pick (#14834)
* Use DEFAULT macro in C APIs (#14767) (#14789)
* Set idx2name for Optimizer object (#14703) (#14772)
* Add pin_device_id option to Gluon DataLoader (#14136) (#14771)
* Tidy up storage allocation and deallocation (#14480) (#14768)
* Add MXEnginePushAsync and MXEnginePushSync C APIs (#14615) (#14770)
* Less cudaGet/SetDevice calls in Gluon execution (#13764)
* Fix nightly build of 1.4.x (#14556)
* Memory fixes. Resolves #10867, and resolves #14080 (#14372) (#14586)
* Fixes for data links (#14526)
* Backport of Windows CI Fixes (#14420)


## 1.4.0

- [New Features](#new-features-2)
  * [Java Inference API](#java-inference-api)
  * [Julia API](#julia-api)
  * [Control Flow Operators (experimental)](#control-flow-operators-experimental)
  * [SVRG Optimization](#svrg-optimization)
  * [Subgraph API (experimental)](#subgraph-api-experimental)
  * [JVM Memory Management](#jvm-memory-management)
  * [Topology-aware AllReduce (experimental)](#topology-aware-allreduce-experimental)
  * [MKLDNN backend: Graph optimization and Quantization (experimental)](#mkldnn-backend--graph-optimization-and-quantization-experimental)
    + [Graph Optimization](#graph-optimization)
    + [Quantization](#quantization)
- [New Operators](#new-operators-3)
- [Feature improvements](#feature-improvements-3)
  * [Operator](#operator)
  * [Optimizer](#optimizer)
  * [Sparse](#sparse)
  * [ONNX](#onnx)
  * [MKLDNN](#mkldnn-2)
  * [Inference](#inference)
  * [Other](#other)
- [Frontend API updates](#frontend-api-updates)
  * [Gluon](#gluon-2)
  * [Symbol](#symbol-1)
- [Language API updates](#language-api-updates)
  * [Java](#java)
  * [R](#r)
  * [Scala](#scala-2)
  * [Clojure](#clojure-2)
  * [Perl](#perl-2)
  * [Julia](#julia-2)
- [Performance benchmarks and improvements](#performance-benchmarks-and-improvements)
- [Bug fixes](#bug-fixes-4)
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
    + [TVM:](#tvm)
    + [CUDNN:](#cudnn)
    + [Horovod:](#horovod)
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
Note: This is an experimental feature and has known problems - see [13341](https://github.com/apache/mxnet/issues/13341). Please help to contribute to improve the robustness of the feature.

#### MKLDNN backend: Graph optimization and Quantization (experimental)

Two advanced features, graph optimization (operator fusion) and reduced-precision (INT8) computation, are introduced to MKLDNN backend in this release ([#12530](https://github.com/apache/mxnet/pull/12530), [#13297](https://github.com/apache/mxnet/pull/13297), [#13260](https://github.com/apache/mxnet/pull/13260)).
These features significantly boost the inference performance on CPU (up to 4X) for a broad range of deep learning topologies. Currently, this feature is only available for inference on platforms with [supported Intel CPUs](https://github.com/intel/mkl-dnn#system-requirements).

##### Graph Optimization
MKLDNN backend takes advantage of MXNet subgraph to implement the most of possible operator fusions for inference, such as Convolution + ReLU, Batch Normalization folding, etc. When using mxnet-mkl package, users can easily enable this feature by setting export MXNET_SUBGRAPH_BACKEND=MKLDNN.

##### Quantization
Performance of reduced-precision (INT8) computation is also dramatically improved after the graph optimization feature is applied on CPU Platforms. Various models are supported and can benefit from reduced-precision computation, including symbolic models, Gluon models and even custom models. Users can run most of the pre-trained models with only a few lines of commands and a new quantization script imagenet_gen_qsym_mkldnn.py. The observed accuracy loss is less than 0.5% for popular CNN networks, like ResNet-50, Inception-BN, MobileNet, etc.

Please find detailed information and performance/accuracy numbers here: [MKLDNN README](https://mxnet.apache.org/api/python/docs/tutorials/performance/backend/mkldnn/mkldnn_readme.html), [quantization README](https://github.com/apache/mxnet/tree/master/example/quantization#1) and [design proposal](https://cwiki.apache.org/confluence/display/MXNET/MXNet+Graph+Optimization+and+Quantization+based+on+subgraph+and+MKL-DNN)

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
* Import Julia binding (#10149), how to use is available at https://github.com/apache/mxnet/tree/master/julia

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
Please follow the instructions at https://mxnet.apache.org/install/index.html

### List of submodules used by Apache MXNet (Incubating) and when they were updated last
Submodule@commit ID::Last updated by MXNet:: Last update in submodule

* cub@05eb57f::Jul 31, 2017 :: Jul 31, 2017
* dlpack@10892ac:: Oct 30, 2017 :: Aug 23, 2018
* dmlc-core@0a0e8ad:: Aug 15, 2018 :: Nov 15, 2018
* googletest@ec44c6c:: July 14, 2016 :: July 14, 2016
* mkldnn@722901c:: Feb 13, 2019 :: Feb 12, 2019
* mshadow@696803b:: Sep 28, 2018 :: Nov 7,  2018
* onnx-tensorrt@3d8ee04:: Aug 22, 2018 :: Nov 10, 2018
* openmp@37c7212: Nov 22, 2017 :: Nov 13, 2018
* ps-lite@8a76389: April 25, 2018 :: Oct 9, 2018
* tvm@0f053c8: Oct 10, 2018 :: Oct 8, 2018

## 1.3.1

### Bug fixes

* [MXNET-953] Fix oob memory read (v1.3.x) / [#13118](https://github.com/apache/mxnet/pull/13118)
Simple bugfix addressing an out-of-bounds memory read.


* [MXNET-969] Fix buffer overflow in RNNOp (v1.3.x) / [#13119](https://github.com/apache/mxnet/pull/13119)
This fixes an buffer overflow detected by ASAN.


* CudnnFind() usage improvements (v1.3.x) / [#13123](https://github.com/apache/mxnet/pull/13123)
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

* [MXNET-922] Fix memleak in profiler (v1.3.x) / [#13120](https://github.com/apache/mxnet/pull/13120)
  Fix a memleak reported locally by ASAN during a normal inference test.

* Fix lazy record io when used with dataloader and multi_worker > 0 (v1.3.x) / [#13124](https://github.com/apache/mxnet/pull/13124)
  Fixes multi_worker data loader when record file is used. The MXRecordIO instance needs to require a new file handler after fork to be safely manipulated simultaneously.

  This fix also safely voids the previous temporary fixes #12093 #11370.

* fixed symbols naming in RNNCell, LSTMCell, GRUCell (v1.3.x) / [#13158](https://github.com/apache/mxnet/pull/13158)
  This fixes #12783, by assigning all nodes in hybrid_forward a unique name. Some operations were in fact performed without attaching the appropriate (time) prefix to the name, which makes serialized graphs non-deserializable.

* Fixed `__setattr__` method of `_MXClassPropertyMetaClass` (v1.3.x) / [#13157](https://github.com/apache/mxnet/pull/13157)
  Fixed `__setattr__` method

* allow foreach on input with 0 length (v1.3.x) / [#13151](https://github.com/apache/mxnet/pull/13151)
  Fix #12470. With this change, outs shape can be inferred correctly.

* Infer dtype in SymbolBlock import from input symbol (v1.3.x) / [#13117](https://github.com/apache/mxnet/pull/13117)
  Fix for the issue - #11849
  Currently, Gluon symbol block cannot import any symbol with type other than fp32. All the parameters are created as FP32 leading to failure in importing the params when it is of type fp16, fp64 etc,
  In this PR, we infer the type of the symbol being imported and create the Symbol Block Parameters with that inferred type.
  Added the tests

### Documentation fixes

* Document the newly added env variable (v1.3.x) / [#13156](https://github.com/apache/mxnet/pull/13156)
  Document the env variable: MXNET_ENFORCE_DETERMINISM added in PR: [#12992](https://github.com/apache/mxnet/pull/12992)

* fix broken links (v1.3.x) / [#13155](https://github.com/apache/mxnet/pull/13155)
  This PR fixes broken links on the website.

* fix broken Python IO API docs (v1.3.x) / [#13154](https://github.com/apache/mxnet/pull/13154)
  Fixes [#12854: Data Iterators documentation is broken](https://github.com/apache/mxnet/issues/12854)

  This PR manually specifies members of the IO module so that the docs will render as expected. This is workaround in the docs to deal with a bug introduced in the Python code/structure since v1.3.0. See the comments for more info.

  This PR also fixes another issue that may or may not be related. Cross references to same-named entities like name, shape, or type are confusing Sphinx and it seems to just link to whatever it last dealt with that has the same name, and not the current module. To fix this you have to be very specific. Don't use type, use np.type if that's what you want. Otherwise you might end up with mxnet.kvstore.KVStore.type. This is a known Sphinx issue, so it might be something we have to deal with for the time being.

  This is important for any future modules - that they recognize this issue and make efforts to map the params and other elements.

* add/update infer_range docs (v1.3.x) / [#13153](https://github.com/apache/mxnet/pull/13153)
  This PR adds or updates the docs for the infer_range feature.

  Clarifies the param in the C op docs
  Clarifies the param in the Scala symbol docs
  Adds the param for the Scala ndarray docs
  Adds the param for the Python symbol docs
  Adds the param for the Python ndarray docs

### Other Improvements

* [MXNET-1179] Enforce deterministic algorithms in convolution layers (v1.3.x) / [#13152](https://github.com/apache/mxnet/pull/13152)
  Some of the CUDNN convolution algorithms are non-deterministic (see issue #11341). This PR adds an env variable to enforce determinism in the convolution operators. If set to true, only deterministic CUDNN algorithms will be used. If no deterministic algorithm is available, MXNet will error out.


### Submodule updates

* update mshadow (v1.3.x) / [#13122](https://github.com/apache/mxnet/pull/13122)
  Update mshadow for omp acceleration when nvcc is not present

### Known issues

The test test_operator.test_dropout has issues and has been disabled on the branch:

* Disable flaky test test_operator.test_dropout (v1.3.x) / [#13200](https://github.com/apache/mxnet/pull/13200)



For more information and examples, see [full release notes](https://cwiki.apache.org/confluence/x/eZGzBQ)


## 1.3.0

### New Features - Gluon RNN layers are now HybridBlocks
- In this release, Gluon RNN layers such as `gluon.rnn.RNN`, `gluon.rnn.LSTM`, `gluon.rnn.GRU` becomes `HybridBlock`s as part of [gluon.rnn improvements project](https://github.com/apache/mxnet/projects/11) (#11482).
- This is the result of newly available fused RNN operators added for CPU: LSTM([#10104](https://github.com/apache/mxnet/pull/10104)), vanilla RNN([#11399](https://github.com/apache/mxnet/pull/11399)), GRU([#10311](https://github.com/apache/mxnet/pull/10311))
- Now many dynamic networks that are based on Gluon RNN layers can now be completely hybridized, exported, and used in the inference APIs in other language bindings such as R, Scala, etc.

### MKL-DNN improvements
- Introducing more functionality support for MKL-DNN as follows:
  - Added support for more activation functions like, "sigmoid", "tanh", "softrelu". ([#10336](https://github.com/apache/mxnet/pull/10336))
  - Added Debugging functionality: Result check ([#12069](https://github.com/apache/mxnet/pull/12069)) and Backend switch ([#12058](https://github.com/apache/mxnet/pull/12058)).

### New Features - Gluon Model Zoo Pre-trained Models
- Gluon Vision Model Zoo now provides MobileNetV2 pre-trained models (#10879) in addition to
  AlexNet, DenseNet, Inception V3, MobileNetV1, ResNet V1 and V2, SqueezeNet 1.0 and 1.1, and VGG
  pretrained models.
- Updated pre-trained models provide state-of-the-art performance on all resnetv1, resnetv2, and vgg16, vgg19, vgg16_bn, vgg19_bn models (#11327 #11860 #11830).

### New Features - Clojure package (experimental)
- MXNet now supports the Clojure programming language. The MXNet Clojure package brings flexible and efficient GPU computing and state-of-art deep learning to Clojure. It enables you to write seamless tensor/matrix computation with multiple GPUs in Clojure. It also lets you construct and customize the state-of-art deep learning models in Clojure, and apply them to tasks, such as image classification and data science challenges.([#11205](https://github.com/apache/mxnet/pull/11205))
- Checkout examples and API documentation [here](https://mxnet.apache.org/api/clojure/index.html).

### New Features - Synchronized Cross-GPU Batch Norm (experimental)
- Gluon now supports Synchronized Batch Normalization (#11502).
- This enables stable training on large-scale networks with high memory consumption such as FCN for image segmentation.

### New Features - Sparse Tensor Support for Gluon (experimental)
- Sparse gradient support is added to `gluon.nn.Embedding`. Set `sparse_grad=True` to enable when constructing the Embedding block. ([#10924](https://github.com/apache/mxnet/pull/10924))
- Gluon Parameter now supports "row_sparse" storage type, which reduces communication cost and memory consumption for multi-GPU training for large models. `gluon.contrib.nn.SparseEmbedding` is an example empowered by this. ([#11001](https://github.com/apache/mxnet/pull/11001), [#11429](https://github.com/apache/mxnet/pull/11429))
- Gluon HybridBlock now supports hybridization with sparse operators ([#11306](https://github.com/apache/mxnet/pull/11306)).

### New Features - Control flow operators (experimental)
- This is the first step towards optimizing dynamic neural networks with variable computation graphs, by adding symbolic and imperative control flow operators. [Proposal](https://cwiki.apache.org/confluence/display/MXNET/Optimize+dynamic+neural+network+models+with+control+flow+operators).
- New operators introduced: foreach([#11531](https://github.com/apache/mxnet/pull/11531)), while_loop([#11566](https://github.com/apache/mxnet/pull/11566)), cond([#11760](https://github.com/apache/mxnet/pull/11760)).

### New Features - Scala API Improvements (experimental)
- Improvements to MXNet Scala API usability([#10660](https://github.com/apache/mxnet/pull/10660), [#10787](https://github.com/apache/mxnet/pull/10787), [#10991](https://github.com/apache/mxnet/pull/10991))
- Symbol.api and NDArray.api would bring new set of functions that have complete definition for all arguments.
- Please see this [Type safe API design document](https://cwiki.apache.org/confluence/display/MXNET/Scala+Type-safe+API+Design+Doc) for more details.

### New Features - Rounding GPU Memory Pool for dynamic networks with variable-length inputs and outputs (experimental)
- MXNet now supports a new memory pool type for GPU memory (#11041).
- Unlike the default memory pool requires exact size match to reuse released memory chunks, this new memory pool uses exponential-linear rounding so that similar sized memory chunks can all be reused, which is more suitable for all the workloads with dynamic-shape inputs and outputs. Set environment variable `MXNET_GPU_MEM_POOL_TYPE=Round` to enable.

### New Features - Topology-aware AllReduce (experimental)
- This features uses trees to perform the Reduce and Broadcast. It uses the idea of minimum spanning trees to do a binary tree Reduce communication pattern to improve it. This topology aware approach reduces the existing limitations for single machine communication shown by mehods like parameter server and NCCL ring reduction. It is an experimental feature ([#11591](https://github.com/apache/mxnet/pull/11591)).
- Paper followed for implementation: [Optimal message scheduling for aggregation](https://www.sysml.cc/doc/178.pdf).
- Set environment variable `MXNET_KVSTORE_USETREE=1` to enable.

### New Features - Export MXNet models to ONNX format (experimental)
- With this feature, now MXNet models can be exported to ONNX format([#11213](https://github.com/apache/mxnet/pull/11213)). Currently, MXNet supports ONNX v1.2.1. [API documentation](https://mxnet.apache.org/api/python/contrib/onnx.html).
- Checkout this [tutorial](https://mxnet.apache.org/tutorials/onnx/export_mxnet_to_onnx.html) which shows how to use MXNet to ONNX exporter APIs. ONNX protobuf so that those models can be imported in other frameworks for inference.

### New Features - TensorRT Runtime Integration (experimental)
- [TensorRT](https://developer.nvidia.com/tensorrt) provides significant acceleration of model inference on NVIDIA GPUs compared to running the full graph in MxNet using unfused GPU operators. In addition to faster fp32 inference, TensorRT optimizes fp16 inference, and is capable of int8 inference (provided the quantization steps are performed). Besides increasing throughput, TensorRT significantly reduces inference latency, especially for small batches.
- This feature in MXNet now introduces runtime integration of TensorRT into MXNet, in order to accelerate inference.([#11325](https://github.com/apache/mxnet/pull/11325))
- Currently, its in contrib package.

### New Examples - Scala
- Refurnished Scala Examples with improved API, documentation and CI test coverage. ([#11753](https://github.com/apache/mxnet/pull/11753), [#11621](https://github.com/apache/mxnet/pull/11621) )
- Now all Scala examples have:
  - No bugs block in the middle
  - Good Readme to start with
  - with Type-safe API usage inside
  - monitored in CI in each PR runs

### Maintenance - Flaky Tests improvement effort
- Fixed 130 flaky tests on CI. Tracked progress of the project [here](https://github.com/apache/mxnet/projects/9).
- Add flakiness checker (#11572)

### Maintenance - MXNet Model Backwards Compatibility Checker
- This tool ([#11626](https://github.com/apache/mxnet/pull/11626)) helps in ensuring consistency and sanity while performing inference on the latest version of MXNet using models trained on older versions of MXNet.
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
- Support for dot(dns, csr) = dns and dot(dns, csr.T) = dns on CPU ([#11113](https://github.com/apache/mxnet/pull/11113))
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
- Implemented a new ONNX module in MXNet which offers an easy to use API to import ONNX models into MXNet's symbolic interface (#9963). Checkout the [example](https://github.com/apache/mxnet/blob/master/example/onnx/super_resolution.py) on how you could use this [API](https://cwiki.apache.org/confluence/display/MXNET/ONNX-MXNet+API+Design) to import ONNX models and perform inference on MXNet. Currently, the ONNX-MXNet Import module is still experimental. Please use it with caution.

### New Features - Added Support for Model Quantization with Calibration
- Implemented model quantization by adopting the [TensorFlow approach](https://www.tensorflow.org/performance/quantization) with calibration by borrowing the idea from Nvidia's [TensorRT](http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf). The focus of this work is on keeping quantized models (ConvNets for now) inference accuracy loss under control when compared to their corresponding FP32 models. Please see the [example](https://github.com/apache/mxnet/tree/master/example/quantization) on how to quantize a FP32 model with or without calibration (#9552). Currently, the Quantization support is still experimental. Please use it with caution.

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
- Fixed tests - Flakiness/Bugs - (#9598, #9951, #10259, #10197, #10136, #10422). Please see: [Tests Improvement Project](https://github.com/apache/mxnet/projects/9)
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
- [Tutorial on fine-tuning an ONNX model](https://github.com/apache/mxnet/blob/v1.2.0/docs/tutorials/onnx/fine_tuning_gluon.md) fails when using cpu context.
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
  - MXNet now compiles and runs on ARMv6, ARMv7, ARMv64 including Raspberry Pi devices. See https://github.com/apache/mxnet/tree/master/docker_multiarch for more information.
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
  - Fixed various bugs in Faster-RCNN example: https://github.com/apache/mxnet/pull/6486
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
  - [Neural art](https://github.com/apache/mxnet/tree/v0.7.0/example/neural-style) by @antinucleon
  - [Detection](https://github.com/apache/mxnet/tree/v0.7.0/example/rcnn), RCNN bt @precedenceguo
  - [Segmentation](https://github.com/apache/mxnet/tree/v0.7.0/example/fcn-xs), FCN by @tornadomeet
  - [Face identification](https://github.com/tornadomeet/mxnet-face) by @tornadomeet
  - More on the example

## v0.5 (initial release)
- All basic modules ready
