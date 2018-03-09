Environment Variables
=====================
MXNet has several settings that you can change with environment variables.
Typically, you wouldn't need to change these settings, but they are listed here for reference.

For example, you can set these environment variables in Linux or macOS as follows:
```
export MXNET_GPU_WORKER_NTHREADS=3
```

## Set the Number of Threads

* MXNET_GPU_WORKER_NTHREADS
  - Values: Int ```(default=2)```
  - The maximum number of threads to use on each GPU. This parameter is used to parallelize the computation within a single GPU card.
* MXNET_GPU_COPY_NTHREADS
  - Values: Int ```(default=1)```
  - The maximum number of concurrent threads that do the memory copy job on each GPU.
* MXNET_CPU_WORKER_NTHREADS
  - Values: Int ```(default=1)```
  - The maximum number of scheduling threads on CPU. It specifies how many operators can be run in parallel.
* MXNET_CPU_PRIORITY_NTHREADS
  - Values: Int ```(default=4)```
  - The number of threads given to prioritized CPU jobs.
* MXNET_CPU_NNPACK_NTHREADS
  - Values: Int ```(default=4)```
  - The number of threads used for NNPACK. NNPACK package aims to provide high-performance implementations of some layers for multi-core CPUs. Checkout [NNPACK](http://mxnet.io/faq/nnpack.html) to know more about it.

## Memory Options

* MXNET_EXEC_ENABLE_INPLACE
  - Values: true or false ```(default=true)```
  - Whether to enable in-place optimization in symbolic execution. Checkout [in-place optimization](http://mxnet.io/architecture/note_memory.html#in-place-operations) to know more about it.
* NNVM_EXEC_MATCH_RANGE
  - Values: Int ```(default=16)```
  - The approximate matching scale in the symbolic execution memory allocator.
  - Set this to 0 if you don't want to enable memory sharing between graph nodes(for debugging purposes).
  - This variable has impact on the result of memory planning. So, MXNet sweep between [1, NNVM_EXEC_MATCH_RANGE], and selects the best value.
* MXNET_EXEC_NUM_TEMP
  - Values: Int ```(default=1)```
  - The maximum number of temporary workspaces to allocate to each device. This controls space replicas and in turn reduces the memory usage.
  - Setting this to a small number can save GPU memory. It will also likely decrease the level of parallelism, which is usually acceptable.
  - MXNet internally uses graph coloring algorithm to [optimize memory consumption](http://mxnet.io/architecture/note_memory.html).
  - This parameter is also used to get number of matching colors in graph and in turn how much parallelism one can get in each GPU. Color based match usually costs more memory but also enables more parallelism.
* MXNET_GPU_MEM_POOL_RESERVE
  - Values: Int ```(default=5)```
  - The percentage of GPU memory to reserve for things other than the GPU array, such as kernel launch or cudnn handle space.
  - If you see a strange out-of-memory error from the kernel launch, after multiple iterations, try setting this to a larger value.  

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
  - The maximum number of nodes in the subgraph executed in bulk during training(not inference). Setting this to a larger number may reduce the degree of parallelism for multi-GPU training.

## Control the Data Communication

* MXNET_KVSTORE_REDUCTION_NTHREADS
  - Values: Int ```(default=4)```
	- The number of CPU threads used for summing big arrays.
* MXNET_KVSTORE_BIGARRAY_BOUND
  - Values: Int ```(default=1000000)```
  - The minimum size of a "big array".
  - When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads are used for reduction.
  - This parameter is also used as a load balancer in kvstore. It controls when to partition a single weight to all the servers. If the size of a single weight is less than MXNET_KVSTORE_BIGARRAY_BOUND then, it is sent to a single randomly picked server otherwise it is partitioned to all the servers.
* MXNET_ENABLE_GPU_P2P
  - Values: 0(false) or 1(true) ```(default=1)```
  - If true, MXNet tries to use GPU peer-to-peer communication, if available on your device,
    when kvstore's type is `device`.

## Memonger

* MXNET_BACKWARD_DO_MIRROR
  - Values: 0(false) or 1(true) ```(default=0)```
  - MXNet uses mirroring concept to save memory. Normally backward pass needs some forward input and it is stored in memory but you can choose to release this saved input and recalculate it in backward pass when needed. This basically trades off the computation for memory consumption.
  - This parameter decides whether to do `mirror` during training for saving device memory.
  - When set to `1`, during forward propagation, graph executor will `mirror` some layer's feature map and drop others, but it will re-compute this dropped feature maps when needed.
  - `MXNET_BACKWARD_DO_MIRROR=1` will save 30%~50% of device memory, but retains about 95% of running speed.
  - One extension of `mirror` in MXNet is called [memonger technology](https://arxiv.org/abs/1604.06174), it will only use O(sqrt(N)) memory at 75% running speed. Checkout the code [here](https://github.com/dmlc/mxnet-memonger).

## Control the profiler

When USE_PROFILER is enabled in Makefile or CMake, the following environments can be used to profile the application without changing code. Execution options may affect the granularity of profiling result. If you need profiling result of every operator, please set MXNET_EXEC_BULK_EXEC_INFERENCE and MXNET_EXEC_BULK_EXEC_TRAIN to 0.

* MXNET_PROFILER_AUTOSTART
  - Values: 0(false) or 1(true) ```(default=0)```
	- Set to 1, MXNet starts the profiler automatically. The profiling result is stored into profile.json in the working directory.

* MXNET_PROFILER_MODE
  - Values: 0(false) or 1(true) ```(default=0)```
	- If set to '0', profiler records the events of the symbolic operators.
	- If set to '1', profiler records the events of all operators.

## Other Environment Variables

* MXNET_CUDNN_AUTOTUNE_DEFAULT
  - Values: 0(false) or 1(true) ```(default=1)```
  - The default value of cudnn auto tunning for convolution layers.
  - Auto tuning is turned off by default. For benchmarking, set this to 1 to turn it on by default.

* MXNET_GLUON_REPO
  - Values: String ```(default='https://apache-mxnet.s3-accelerate.dualstack.amazonaws.com/'```
  - The repository url to be used for Gluon datasets and pre-trained models.

Settings for Minimum Memory Usage
---------------------------------
- Make sure ```min(MXNET_EXEC_NUM_TEMP, MXNET_GPU_WORKER_NTHREADS) = 1```
  - The default setting satisfies this.

Settings for More GPU Parallelism
---------------------------------
- Set ```MXNET_GPU_WORKER_NTHREADS``` to a larger number (e.g., 2)
  - To reduce memory usage, consider setting ```MXNET_EXEC_NUM_TEMP```.
  - This might not speed things up, especially for image applications, because GPU is usually fully utilized even with serialized jobs.
