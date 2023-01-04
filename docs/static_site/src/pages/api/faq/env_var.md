---
layout: page_category
title:  Environment Variables
category: faq
faq_c: Deployment Environments
question: What are MXNet environment variables?
permalink: /api/faq/env_var
---
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

Environment Variables
=====================
MXNet has several settings that you can change with environment variables.
Typically, you wouldn't need to change these settings, but they are listed here for reference.

For example, you can set these environment variables in Linux or macOS as follows:
```
export MXNET_GPU_WORKER_NTHREADS=3
```

Or in powershell:
```
$env:MXNET_STORAGE_FALLBACK_LOG_VERBOSE=0
```

## Variables controlling the execution environment

* MXNET_LIBRARY_PATH
    Absolute path indicating where the mxnet dynamic library is to be located, this would be the absolute
    path to `libmxnet.so` or `libmxnet.dll` depending on the platform. The logic for loading the
    library is in `python/mxnet/libinfo.py`

## Set the Number of Threads

* MXNET_GPU_WORKER_NTHREADS
  - Values: Int ```(default=2)```
  - The maximum number of threads to use on each GPU. This parameter is used to parallelize the computation within a single GPU card.
* MXNET_GPU_COPY_NTHREADS
  - Values: Int ```(default=2)```
  - The maximum number of concurrent threads that do the memory copy job on each GPU.
* MXNET_CPU_WORKER_NTHREADS
  - Values: Int ```(default=1)```
  - The maximum number of scheduling threads on CPU. It specifies how many operators can be run in parallel. Note that most CPU operators are parallelized by OpenMP. To change the number of threads used by individual operators, please set `OMP_NUM_THREADS` instead.
* MXNET_CPU_PRIORITY_NTHREADS
  - Values: Int ```(default=4)```
  - The number of threads given to prioritized CPU jobs.
* MXNET_MP_WORKER_NTHREADS
  - Values: Int ```(default=1)```
  - The number of scheduling threads on CPU given to multiprocess workers. Enlarge this number allows more operators to run in parallel in individual workers but please consider reducing the overall `num_workers` to avoid thread contention (not available on Windows).
* MXNET_MP_OPENCV_NUM_THREADS
  - Values: Int ```(default=0)```
  - The number of OpenCV execution threads given to multiprocess workers. OpenCV multithreading is disabled if `MXNET_MP_OPENCV_NUM_THREADS` < 1 (default). Enlarge this number may boost the performance of individual workers when executing underlying OpenCV functions but please consider reducing the overall `num_workers` to avoid thread contention (not available on Windows).

## Memory Options

* MXNET_EXEC_ENABLE_INPLACE
  - Values: true or false ```(default=true)```
    - Whether to enable in-place optimization in symbolic execution. Checkout [in-place optimization]({{'/api/architecture/note_memory#in-place-operations'|relative_url}}) to know more about it.
* NNVM_EXEC_MATCH_RANGE
  - Values: Int ```(default=16)```
  - The approximate matching scale in the symbolic execution memory allocator.
  - Set this to 0 if you don't want to enable memory sharing between graph nodes(for debugging purposes).
  - This variable has impact on the result of memory planning. So, MXNet sweep between [1, NNVM_EXEC_MATCH_RANGE], and selects the best value.
* MXNET_EXEC_NUM_TEMP
  - Values: Int ```(default=1)```
  - The maximum number of temporary workspaces to allocate to each device. This controls space replicas and in turn reduces the memory usage.
  - Setting this to a small number can save GPU memory. It will also likely decrease the level of parallelism, which is usually acceptable.
    - MXNet internally uses graph coloring algorithm to [optimize memory consumption]({{'/api/architecture/note_memory'|relative_url}}).
  - This parameter is also used to get number of matching colors in graph and in turn how much parallelism one can get in each GPU. Color based match usually costs more memory but also enables more parallelism.
* MXNET_GPU_MEM_POOL_TYPE
  - Values: String ```(default=Naive)```
  - The type of GPU memory pool.
  - Choices:
    - *Naive*: A simple memory pool that allocates memory for the requested size and cache memory buffers, when this memory is released. The size of memory chunk is defined by rounding the requested memory size to the nearest bigger multiple of MXNET_GPU_MEM_POOL_PAGE_SIZE (or MXNET_GPU_MEM_LARGE_ALLOC_ROUND_SIZE, when the result of rounding for MXNET_GPU_MEM_POOL_PAGE_SIZE is bigger than MXNET_GPU_MEM_LARGE_ALLOC_ROUND_SIZE) and allocates memory of the rounded size.
    - *Round*: A memory pool that try to rounds the requested memory size to the nearest bigger power of 2. When this rounded number is bigger that 2**MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF, the *Naive* rounding algorithm is used. Caching and allocating buffered memory works in the same way as the naive memory pool.
    - *Unpooled*: No memory pool is used.
* MXNET_GPU_MEM_POOL_RESERVE
  - Values: Int ```(default=5)```
  - The percentage of GPU memory to reserve for things other than the GPU array, such as kernel launch or cudnn handle space.
  - The value is used only by the GPU memory pool. If it is not possible to allocate new memory AND still save this reserve, the memory pool will free the cached memory.
  - If you see a strange out-of-memory error from the kernel launch, after multiple iterations, try setting this to a larger value.
* MXNET_GPU_MEM_LARGE_ALLOC_ROUND_SIZE
  - Values: Int ```(default=2097152)```
  - When the rounded size of memory allocations calculated by the pool of *Naive* type is larger than this threshold, it will be rounded up to a multiple of this value.
  - The default was chosen to minimize global memory fragmentation within the GPU driver. Set this to 1 to disable.
* MXNET_GPU_MEM_POOL_ROUND_LINEAR_CUTOFF
  - Values: Int ```(default=24)```
  - The cutoff threshold used by *Round* strategy. Let's denote the threshold as T. If the memory size is smaller than `2 ** T` (by default, it's 2 ** 24 = 16MB), it rounds to the smallest `2 ** n` that is larger than the requested memory size; if the memory size is larger than `2 ** T`, it rounds to the next k * 2 ** T.
* MXNET_CPU_MEM_POOL_TYPE
  - Values: String ```(default=Naive)```
  - The type of CPU memory pool.
  - Choices:
    - *Naive*: A simple memory pool that allocates memory for the requested size and cache memory buffers, when this memory is released. The size of memory chunk is defined by rounding the requested memory size to the nearest bigger multiple of MXNET_CPU_MEM_POOL_PAGE_SIZE (or MXNET_CPU_MEM_LARGE_ALLOC_ROUND_SIZE, when the result of rounding for MXNET_CPU_MEM_POOL_PAGE_SIZE is bigger than MXNET_CPU_MEM_LARGE_ALLOC_ROUND_SIZE) and allocates memory of the rounded size.
    - *Round*: A memory pool that try to rounds the requested memory size to the nearest bigger power of 2. When this rounded number is bigger that 2**MXNET_CPU_MEM_POOL_ROUND_LINEAR_CUTOFF, the the *Naive* rounding algorithm is used. Caching and allocating buffered memory works in the same way as the naive memory pool.
    - *Unpooled*: No memory pool is used.
* MXNET_CPU_MEM_POOL_RESERVE
  - Values: Int ```(default=5)```
  - The percentage of CPU memory to reserve for things other than the CPU array.
  - The value is used only by the CPU memory pool. If it is not possible to allocate new memory AND still save this reserve, the memory pool will free the cached memory.
  - If you see a strange out-of-memory error from the kernel launch, after multiple iterations, try setting this to a larger value.
* MXNET_CPU_MEM_LARGE_ALLOC_ROUND_SIZE
  - Values: Int ```(default=2097152)```
  - When the rounded size of memory allocations calculated by the pool of *Naive* type is larger than this threshold, it will be rounded up to a multiple of this value.
  - Set this to 1 to disable.
* MXNET_CPU_MEM_POOL_ROUND_LINEAR_CUTOFF
  - Values: Int ```(default=24)```
  - The cutoff threshold used by *Round* strategy. Let's denote the threshold as T. If the memory size is smaller than `2 ** T` (by default, it's 2 ** 24 = 16MB), it rounds to the smallest `2 ** n` that is larger than the requested memory size; if the memory size is larger than `2 ** T`, it rounds to the next k * 2 ** T.
* MXNET_CPU_PINNED_MEM_POOL_TYPE
  - Values: String ```(default=Naive)```
  - The type of CPU_PINNED memory pool.
  - Choices:
    - *Naive*: A simple memory pool that allocates memory for the requested size and cache memory buffers, when this memory is released. The size of memory chunk is defined by rounding the requested memory size to the nearest bigger multiple of MXNET_CPU_PINNED_MEM_POOL_PAGE_SIZE (or MXNET_CPU_PINNED_MEM_LARGE_ALLOC_ROUND_SIZE, when the result of rounding for MXNET_CPU_PINNED_MEM_POOL_PAGE_SIZE is bigger than MXNET_CPU_PINNED_MEM_LARGE_ALLOC_ROUND_SIZE) and allocates memory of the rounded size.
    - *Round*: A memory pool that try to rounds the requested memory size to the nearest bigger power of 2. When this rounded number is bigger that 2**MXNET_CPU_PINNED_MEM_POOL_ROUND_LINEAR_CUTOFF, the the *Naive* rounding algorithm is used. Caching and allocating buffered memory works in the same way as the naive memory pool.
    - *Unpooled*: No memory pool is used.
* MXNET_CPU_PINNED_MEM_POOL_RESERVE
  - Values: Int ```(default=5)```
  - The percentage of GPU memory to reserve for things other than the GPU array.
  - The value is used only by the CPU memory pool. If it is not possible to allocate new memory AND still save this reserve, the memory pool will free the cached memory.
  - If you see a strange out-of-memory error from the kernel launch, after multiple iterations, try setting this to a larger value.
* MXNET_CPU_PINNED_MEM_LARGE_ALLOC_ROUND_SIZE
  - Values: Int ```(default=2097152)```
  - When the rounded size of memory allocations calculated by the pool of *Naive* type is larger than this threshold, it will be rounded up to a multiple of this value.
  - Set this to 1 to disable.
* MXNET_CPU_PINNED_MEM_POOL_ROUND_LINEAR_CUTOFF
  - Values: Int ```(default=24)```
  - The cutoff threshold used by *Round* strategy. Let's denote the threshold as T. If the memory size is smaller than `2 ** T` (by default, it's 2 ** 24 = 16MB), it rounds to the smallest `2 ** n` that is larger than the requested memory size; if the memory size is larger than `2 ** T`, it rounds to the next k * 2 ** T.
* MXNET_USE_NAIVE_STORAGE_MANAGERS
  - Values: Int ```(default=0)```
  - When value is not 0, no memory pools will be used for any of the following three types of memory: GPU, CPU, CPU_PINNED.
   
## Engine Type

* MXNET_ENGINE_TYPE
  - Values: String ```(default=ThreadedEnginePerDevice)```
  - The type of underlying execution engine of MXNet.
  - Choices:
    - NaiveEngine: A very simple engine that uses the master thread to do the computation synchronously. Setting this engine disables multi-threading. You can use this type for debugging in case of any error. Backtrace will give you the series of calls that lead to the error. Remember to set MXNET_ENGINE_TYPE back to empty after debugging.
    - ThreadedEngine: A threaded engine that uses a global thread pool to schedule jobs.
    - ThreadedEnginePerDevice: A threaded engine that allocates thread per GPU and executes jobs asynchronously.

## Execution Options

* MXNET_EXEC_BULK_EXEC_INFERENCE
  - Values: 0(false) or 1(true) ```(default=1)```
  - If set to `1`, during inference MXNet executes the entire computation graph in bulk mode, which reduces kernel launch gaps in between symbolic operators.
* MXNET_EXEC_BULK_EXEC_TRAIN
  - Values: 0(false) or 1(true) ```(default=1)```
  - If set to `1`, during training MXNet executes the computation graph as several subgraphs in bulk mode.
* MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN
  - Values: Int ```(default=15)```
  - The maximum number of nodes in the subgraph executed in bulk during training (not inference). Setting this to a larger number may reduce the degree of parallelism for multi-GPU training.
* MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD
  - Values: Int ```(default=<value of MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN>)```
  - The maximum number of nodes in the subgraph executed in bulk during training (not inference) in the forward pass.
* MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD
  - Values: Int ```(default=<value of MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN>)```
  - The maximum number of nodes in the subgraph executed in bulk during training (not inference) in the backward pass.
* MXNET_ENABLE_CUDA_GRAPHS
  - Values: 0(false) or 1(true) ```(default=0)```
  - If set to `1`, MXNet will utilize CUDA graphs when executing models on the GPU when possible.
  - For CUDA graphs execution, one needs to use either symbolic model or Gluon model hybridized with options `static_alloc` and `static_shape` set to True.
* MXNET_CUDA_GRAPHS_VERBOSE
  - Values: 0(false) or  1(true) ```(default=0)```
  - If set to `1`, CUDA graphs executor will provide information about the graph being captured and executed.
* MXNET_CUDA_GRAPHS_MAX_LOG_ENTRIES
  - Values: Int ```(default=0)```
  - The maximum number of log messages generated by CUDA graphs executor.
* MXNET_CUDA_GRAPHS_DBG_FILE
  - Values: String ```(default='', to indicate no debug dot files should be created)```
  - The file prefix for '.dot' files for each graph created.  Full path is <prefix>-devN-{trn,inf}.<graphId>.dot .
* MXNET_CUDA_GRAPHS_DBG_FILE_FLAGS
  - Values: Int ```(default=<most verbose setting- includes all info>)```
  - A bitmask to enable various types of info in the debug '.dot' files.  See cudaGraphDebugDotFlags in the CUDA runtime API doc for details.

## Control the Data Communication

* MXNET_KVSTORE_REDUCTION_NTHREADS
  - Values: Int ```(default=4)```
  - The number of CPU threads used for summing up big arrays on a single machine
  - This will also be used for `dist_sync` kvstore to sum up arrays from different contexts on a single machine.
  - This does not affect summing up of arrays from different machines on servers.
  - Summing up of arrays for `dist_sync_device` kvstore is also unaffected as that happens on GPUs.

* MXNET_KVSTORE_BIGARRAY_BOUND
  - Values: Int ```(default=1000000)```
  - The minimum size of a "big array".
  - When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads are used for reduction.
  - This parameter is also used as a load balancer in kvstore. It controls when to partition a single weight to all the servers. If the size of a single weight is less than MXNET_KVSTORE_BIGARRAY_BOUND then, it is sent to a single randomly picked server otherwise it is partitioned to all the servers.

* MXNET_KVSTORE_USETREE
  - Values: 0(false) or 1(true) ```(default=0)```
  - If true, MXNet tries to use tree reduction for Push and Pull communication.
  - Otherwise, MXNet uses the default Push and Pull implementation.
  - Tree reduction technology has been shown to be faster than the standard ```--kv-store device``` Push/Pull and ```--kv-store nccl``` Push/Pull for small batch sizes.

* MXNET_KVSTORE_LOGTREE
  - Values: 0(false) or 1(true) ```(default=0)```
  - If true and MXNET_KVSTORE_USETREE is set to 1, MXNet will log the reduction trees that have been generated.

* MXNET_KVSTORE_TREE_ARRAY_BOUND
  - Values: Int ```(default=10000000)```
  - The minimum size of a "big array".
  - When the array size is bigger than this threshold and MXNET_KVSTORE_USETREE is set to 1, multiple trees are used to load balance the big gradient being communicated in order to better saturate link bandwidth.
  - Note: This environmental variable only takes effect if Tree KVStore is being used (MXNET_KVSTORE_USETREE=1).

* MXNET_KVSTORE_TREE_BACKTRACK
  - Values: 0(false) or 1(true) ```(default=0)
  - If true and MXNET_KVSTORE_USETREE is set to 1, MXNet tries to use backtracking to generate the trees required for tree reduction.
  - If false and MXNET_KVSTORE_USETREE is set to 1, MXNet tries to use Kernighan-Lin heuristic to generate the trees required for tree reduction.

* MXNET_KVSTORE_TREE_LINK_USAGE_PENALTY
  - Values: Float ```(default=0.7)```
  - The multiplicative penalty term to a link being used once.

* MXNET_ENABLE_GPU_P2P
  - Values: 0(false) or 1(true) ```(default=1)```
  - If true, MXNet tries to use GPU peer-to-peer communication, if available on your device,
    when kvstore's type is `device`.

* MXNET_UPDATE_ON_KVSTORE
  - Values: 0(false) or 1(true) ```(default=1)```
  - If true, weight updates are performed during the communication step, if possible.

* MXNET_KVSTORE_SLICE_THRESHOLD
  - Values: Int ```(default=40000)```
  - The maximum size of an NDArray slice in terms of number of parameters.
  - This parameter is used to slice an NDArray before synchronizing through P3Store (dist_p3).

## Memory Optimizations

* MXNET_BACKWARD_DO_MIRROR
  - Values: 0(false) or 1(true) ```(default=0)```
  - MXNet uses mirroring concept to save memory. Normally backward pass needs some forward input and it is stored in memory but you can choose to release this saved input and recalculate it in backward pass when needed. This basically trades off the computation for memory consumption.
  - This parameter decides whether to do `mirror` during training for saving device memory.
  - When set to `1`, during forward propagation, graph executor will `mirror` some layer's feature map and drop others, but it will re-compute this dropped feature maps when needed.
  - `MXNET_BACKWARD_DO_MIRROR=1` will save 30%~50% of device memory, but retains about 95% of running speed.
  - One extension of `mirror` in MXNet is called [memonger technology](https://arxiv.org/abs/1604.06174), it will only use O(sqrt(N)) memory at 75% running speed. Checkout the code [here](https://github.com/dmlc/mxnet-memonger).

* MXNET_MEMORY_OPT
  - Values: 0(no optimizations) or 1(highest optimization level) ```(default=0)```
  - If set to '1', various optimizations on memory consumption will be enabled.

## Control the profiler

The following environments can be used to profile the application without changing code. Execution options may affect the granularity of profiling result. If you need profiling result of every operator, please set `MXNET_EXEC_BULK_EXEC_INFERENCE`, `MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN` and `MXNET_EXEC_BULK_EXEC_TRAIN` to 0.

* MXNET_PROFILER_AUTOSTART
  - Values: 0(false) or 1(true) ```(default=0)```
  - Set to 1, MXNet starts the profiler automatically. The profiling result is stored into profile.json in the working directory.

* MXNET_PROFILER_MODE
  - Values: 0(false) to 15(profile everything) ```(default=13)```
  - If set to '0', turns off all profiling.
  - If set to '1', profiler records the events of symbolic operators.
  - If set to '2', profiler records the events of imperative operators.
  - If set to '4', profiler records the C API events.
  - If set to '8', profiler records the events of memory (i.e. storage alloc and free calls).
  - You need to sum the values above for a custom combination. For example, for symbolic and imperative operators, set ```MXNET_PROFILER_MODE=3```(2 + 1).
  - If set to '15', profiler records all the above listed events (API, Memory, Symbolic, Imperative).

## Interface between Python and the C API

* MXNET_ENABLE_CYTHON
  - Values: 0(false), 1(true) ```(default=1)```
  - If set to 0, MXNet uses the ctypes to interface with the C API.
  - If set to 1, MXNet tries to use the cython modules for the ndarray and symbol. If it fails, the ctypes is used or an error occurs depending on MXNET_ENFORCE_CYTHON.

* MXNET_ENFORCE_CYTHON
  - Values: 0(false) or 1(true) ```(default=0)```
  - This has an effect only if MXNET_ENABLE_CYTHON is 1.
  - If set to 0, MXNet fallbacks to the ctypes if importing the cython modules fails.
  - If set to 1, MXNet raises an error if importing the cython modules fails.

If cython modules are used, `mx.nd._internal.NDArrayBase` must be `mxnet._cy3.ndarray.NDArrayBase` for python 3 or `mxnet._cy2.ndarray.NDArrayBase` for python 2.
If ctypes is used, it must be `mxnet._ctypes.ndarray.NDArrayBase`.

## Logging

* DMLC_LOG_STACK_TRACE_DEPTH
  - Values: Int ```(default=0)```
  - The depth of stack trace information to log when exception happens.

## Other Environment Variables

* MXNET_GPU_WORKER_NSTREAMS
  - Values: 1, or 2 ```(default=1)```
  - Determines the number of GPU streams available to operators for their functions.
  - Setting this to 2 may yield a modest performance increase, since ops like the cuDNN convolution op can then calculate their data- and weight-gradients in parallel.
  - Setting this to 2 may also increase a model's demand for GPU global memory.

* MXNET_CUDNN_AUTOTUNE_DEFAULT
  - Values: 0, 1, or 2 ```(default=1)```
  - The default value of cudnn auto tuning for convolution layers.
  - Value of 0 means there is no auto tuning to pick the convolution algo
  - Performance tests are run to pick the convolution algo when value is 1 or 2
  - Value of 1 chooses the best algo in a limited workspace
  - Value of 2 chooses the fastest algo whose memory requirements may be larger than the default workspace threshold

* MXNET_CUDNN_HEUR_MODE
  - Values: 0 or 1 (available since cuDNN 8.1) ```(default=1 for cuDNN 8.1 and later, otherwise 0)```
  - Choose cuDNN heuristics mode.
  - If set to '0', use fast decision tree based method.
  - If set to '1', use neural network based method. It generalizes better for unknown or uncommon models.

* MXNET_CUDNN_ALGO_VERBOSE_LEVEL
  - Values: 0, 1, or 2 ```(default=0)```
  - The level of printed output describing the "convolution engine" configurations
  - Value of 0 produces no output
  - Value of 1 outputs for the chosen config the engine number ("algo"), additional parameters ("knobs") and numerical notes
  - Value of 2 outputs the same info as with a '1' setting, but for all configs considered
  The output can be used to develop engine config filtering strategies to modify model behaviors.
  Numerical accuracy may be improved by filtering out configs shown with 'rp', 'w' or 'fft' (i.e. reduced precision, winograd, or fft).
  The configs are output with their list-index, as suggested by cuDNN, and with the chosen config flagged with a '*'.
  If autotuning is enabled (MXNET_CUDNN_AUTOTUNE_DEFAULT != 0), the measured kernel times will be reported.

* MXNET_CUDA_ALLOW_TENSOR_CORE
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows Tensor Core use in CUDA ops.
  - If set to '1', allows Tensor Core use in CUDA ops.
  - This variable can only be set once in a session.
  - Also controls filtering cuDNN engines with CUDNN_NUMERICAL_NOTE_TENSOR_CORE.

* MXNET_CUDA_TENSOR_OP_MATH_ALLOW_CONVERSION
  - 0(false) or 1(true) ```(default=0)```
  - If set to '0', disallows implicit type conversions to Float16 to use Tensor Cores
  - If set to '1', allows CUDA ops like RNN and Convolution to use TensorCores even with Float32 input data by using implicit type casting to Float16. Only has an effect if `MXNET_CUDA_ALLOW_TENSOR_CORE` is `1`.
  - Also controls filtering cuDNN engines with CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS (such engines are disallowed if set to 0).

* MXNET_CUDNN_ALLOW_REDUCED_PRECISION_REDUCTION
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows cuDNN engines with CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION.
  - If set to '1', allows cuDNN engines with CUDNN_NUMERICAL_NOTE_REDUCED_PRECISION_REDUCTION.

* MXNET_CUDNN_ALLOW_FFT
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows cuDNN engines with CUDNN_NUMERICAL_NOTE_FFT.
  - If set to '1', allows cuDNN engines with CUDNN_NUMERICAL_NOTE_FFT.

* MXNET_CUDNN_ALLOW_WINOGRAD
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows cuDNN engines with CUDNN_NUMERICAL_NOTE_WINOGRAD.
  - If set to '1', allows cuDNN engines with CUDNN_NUMERICAL_NOTE_WINOGRAD.

* MXNET_CUDNN_DISABLED_CONV_FWD_ENGINES
  - Comma-separated list of cuDNN convolution forward engine numbers to disable.
  - Normally should be left alone, unless you know what you're doing.

* MXNET_CUDNN_DISABLED_CONV_DGRAD_ENGINES
  - Comma-separated list of cuDNN convolution dgrad engine numbers to disable.
  - Normally should be left alone, unless you know what you're doing.

* MXNET_CUDNN_DISABLED_CONV_WGRAD_ENGINES
  - Comma-separated list of cuDNN convolution wgrad engine numbers to disable.
  - Normally should be left alone, unless you know what you're doing.

* MXNET_CUDA_LIB_CHECKING
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows various runtime checks of the cuda library version and associated warning messages.
  - If set to '1', permits these checks (e.g. compile vs. link mismatch, old version no longer CI-tested)

* MXNET_CUDNN_LIB_CHECKING
  - 0(false) or 1(true) ```(default=1)```
  - If set to '0', disallows various runtime checks of the cuDNN library version and associated warning messages.
  - If set to '1', permits these checks (e.g. compile vs. link mismatch, old version no longer CI-tested)

* MXNET_GLUON_REPO
  - Values: String ```(default='https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'```
  - The repository url to be used for Gluon datasets and pre-trained models.

* MXNET_HOME
  - Data directory in the filesystem for storage, for example when downloading gluon models.
  - Default in *nix is .mxnet APPDATA/mxnet in windows.

* MXNET_ONEDNN_ENABLED
  - Values: 0, 1 ```(default=1)```
  - Flag to enable or disable oneDNN accelerator. On by default.
  - Only applies to mxnet that has been compiled with oneDNN (```pip install mxnet``` or built from source with ```USE_ONEDNN=1```)

* MXNET_ONEDNN_CACHE_NUM
  - Values: Int ```(default=-1)```
  - Flag to set num of elements that oneDNN cache can hold. Default is -1 which means cache size is unbounded. Should only be set if your model has variable input shapes, as cache size may grow unbounded. The number represents the number of items in the cache and is proportional to the number of layers that use oneDNN and different input shape.

* MXNET_ONEDNN_FORCE_FC_AB_FORMAT
  - Values: 0, 1 ```(default=0)```
  - If set to true, FullyConnected will use only AB format for weights, thus MXNet won't use BRGEMM implementation of FC on machines with AVX512-VNNI support which requires special weights format.

* MXNET_ENFORCE_DETERMINISM
  - Values: 0(false) or 1(true) ```(default=0)```
  - If set to true, MXNet will only use deterministic algorithms in forward and backward computation.
  If no such algorithm exists given other constraints, MXNet will error out. This variable affects the choice
  of CUDNN convolution algorithms. Please see [CUDNN developer guide](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html) for more details.
  - Also controls filtering cuDNN engines with CUDNN_NUMERICAL_NOTE_NONDETERMINISTIC (such engines are disallowed if set to 1).

* MXNET_CPU_PARALLEL_SIZE
  - Values: Int ```(default=200000)```
  - The minimum size to call parallel operations by OpenMP for CPU context.
  - When the array size is bigger than or equal to this threshold, the operation implemented by OpenMP is executed with the Recommended OMP Thread Count.
  - When the array size is less than this threshold, the operation is implemented naively in single thread.

* MXNET_OPTIMIZER_AGGREGATION_SIZE
  - Values: Int ```(default=4)```
  - Maximum value is 60.
  - This variable controls how many weights will be updated in a single call to optimizer (for optimizers that support aggregation, currently limited to SGD).

* MXNET_CPU_TEMP_COPY
  - Values: Int ```(default=4)```
  - This variable controls how many temporary memory resources to create for all CPU context for use in operator.

* MXNET_GPU_TEMP_COPY
  - Values: Int ```(default=1)```
  - This variable controls how many temporary memory resources to create for each GPU context for use in operator.

* MXNET_CPU_PARALLEL_RAND_COPY
  - Values: Int ```(default=1)```
  - This variable controls how many parallel random number generator resources to create for all CPU context for use in operator.

* MXNET_GPU_PARALLEL_RAND_COPY
  - Values: Int ```(default=4)```
  - This variable controls how many parallel random number generator resources to create for each GPU context for use in operator.

* MXNET_GPU_CUDNN_DROPOUT_STATE_COPY
  - Values: Int ```(default=4)```
  - This variable controls how many CuDNN dropout state resources to create for each GPU context for use in operator.

* MXNET_SAFE_ACCUMULATION
  - Values: Values: 0(false) or 1(true) ```(default=1)```
  - If this variable is set, the accumulation will enter the safe mode, meaning accumulation is done in a data type of higher precision than
    the input data type, leading to more accurate accumulation results with a possible performance loss and backward compatibility loss.
    For example, when the variable is set to 1(true), if the input data type is float16, then the accumulation will be done
    with float32.
  - Model accuracies do not necessarily improve with this environment variable turned on.

* MXNET_USE_FUSION
  - Values: 0(false) or 1(true) ```(default=1)```
  - If this variable is set, MXNet will try fusing some of the operations (pointwise operations only for now).
  - It works in Symbolic execution as well as in Gluon models hybridized with ```static_alloc=True``` option.
  - Only applies to MXNet that has been compiled with CUDA (```pip install mxnet-cuXX``` or built from source with ```USE_CUDA=1```) and running on GPU.

* MXNET_RTC_VERBOSE
  - Values: 0(false) or 1(true) ```(default=0)```
  - Only applies to MXNet that has been compiled with CUDA.
  - If this variable is set, MXNet will print the code for operators compiled at runtime.

* MXNET_ELIMINATE_COMMON_EXPR
  - Values: 0(false) or 1(true) ```(default=1)```
  - If this variable is set, MXNet will simplify the computation graph, eliminating duplicated operations on the same inputs.

* MXNET_USE_ONEDNN_RNN
  - Values: 0(false) or 1(true) ```(default=1)```
  - This variable controls whether to use the oneDNN backend in fused RNN operator for CPU context. There are two fusion implementations of RNN operator in MXNet. The oneDNN implementation has a better performance than the naive one, but the latter is more stable in the backward operation currently.

* MXNET_FC_TRUE_FP16
  - Values: 0(false) or 1(true) ```(default=0)```
  - If this variable is set to true, MXNet will perform fp16 accumulation when using cuBLAS and input datatype is set to float16. This could increase the speed of the computation, but might result in loss of accuracy. This makes this setting useful mainly for inference usecases.

* MXNET_RNN_USE_WEIGHT_CACHE
  - Values: 0(false) or 1(true) ```(default=0)```
  - If this variable is set, MXNet will ignore the altering of the version of NDArray which is the input parameter of the RNN operator. In Gluon API, there is a `_rnn_param_concat` operator concatenating the weights and bias of RNN into a single parameter tensor that changes the version number. Since the values of the parameters are invariant in inference pass, the RNN operator could ignore the altering of the version to escape much overhead from re-initializing the parameters.

Settings for Minimum Memory Usage
---------------------------------
- Make sure ```min(MXNET_EXEC_NUM_TEMP, MXNET_GPU_WORKER_NTHREADS) = 1```
  - The default setting satisfies this.

Settings for More GPU Parallelism
---------------------------------
- Set ```MXNET_GPU_WORKER_NTHREADS``` to a larger number (e.g., 2)
  - To reduce memory usage, consider setting ```MXNET_EXEC_NUM_TEMP```.
  - This might not speed things up, especially for image applications, because GPU is usually fully utilized even with serialized jobs.

Settings for controlling OMP tuning
---------------------------------
- Set ```MXNET_USE_OPERATOR_TUNING=0``` to disable Operator tuning code which decides whether to use OMP or not for operator
   - Values: String representation of MXNET_ENABLE_OPERATOR_TUNING environment variable
   -            0=disable all
   -            1=enable all
   -            float32, float16, float32=list of types to enable, and disable those not listed
   - refer : https://github.com/apache/mxnet/blob/master/src/operator/operator_tune-inl.h#L444

- Set ```MXNET_USE_NUM_CORES_OPERATOR_TUNING``` to define num_cores to be used by operator tuning code.
  - This reduces operator tuning overhead when there are multiple instances of mxnet running in the system and we know that
    each mxnet will take only partial num_cores available with system.
  - refer: https://github.com/apache/mxnet/pull/13602
