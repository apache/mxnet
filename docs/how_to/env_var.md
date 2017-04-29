Environment Variables
=====================
MXNet has several settings that you can change with environment variables.
Typically, you wouldn't need to change these settings, but they are listed here for reference.

## Set the Number of Threads

* MXNET_GPU_WORKER_NTHREADS (default=2)
  - The maximum number of threads that do the computation job on each GPU.
* MXNET_GPU_COPY_NTHREADS (default=1)
  - The maximum number of threads that do the memory copy job on each GPU.
* MXNET_CPU_WORKER_NTHREADS (default=1)
  - The maximum number of threads that do the CPU computation job.
* MXNET_CPU_PRIORITY_NTHREADS (default=4)
 - The number of threads given to prioritized CPU jobs.
* MXNET_CPU_NNPACK_NTHREADS (default=4)
 - The number of threads used for NNPACK.

## Memory Options

* MXNET_EXEC_ENABLE_INPLACE (default=true)
  - Whether to enable in-place optimization in symbolic execution.
* NNVM_EXEC_MATCH_RANGE (default=16)
  - The rough matching scale in the symbolic execution memory allocator.
  - Set this to 0 if you don't want to enable memory sharing between graph nodes(for debugging purposes).
* MXNET_EXEC_NUM_TEMP (default=1)
  - The maximum number of temp workspaces to allocate to each device.
  - Setting this to a small number can save GPU memory. It will also likely decrease the level of parallelism, which is usually acceptable.
* MXNET_GPU_MEM_POOL_RESERVE (default=5)
  - The percentage of GPU memory to reserve for things other than the GPU array, such as kernel launch or cudnn handle space.
  - If you see a strange out-of-memory error from the kernel launch, after multiple iterations, try setting this to a larger value.  

## Engine Type

* MXNET_ENGINE_TYPE (default=ThreadedEnginePerDevice)
  - The type of underlying execution engine of MXNet.
  - Choices:
    - NaiveEngine: A very simple engine that uses the master thread to do computation.
    - ThreadedEngine: A threaded engine that uses a global thread pool to schedule jobs.
    - ThreadedEnginePerDevice: A threaded engine that allocates thread per GPU.

## Execution Options

* MXNET_EXEC_BULK_EXEC_INFERENCE (default=1)
  - If set to `1`, during inference MXNet executes the entire computation graph in bulk mode, which reduces kernel launch gaps in between symbolic operators.
* MXNET_EXEC_BULK_EXEC_TRAIN (default=1)
  - If set to `1`, during training MXNet executes the computation graph as several subgraphs in bulk mode.
* MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN (default=15)
  - The maximum number of nodes in the subgraph executed in bulk during training(not inference). Setting this to a larger number may reduce the degree of parallelism for multi-GPU training.

## Control the Data Communication

* MXNET_KVSTORE_REDUCTION_NTHREADS (default=4)
	- The number of CPU threads used for summing big arrays.
* MXNET_KVSTORE_BIGARRAY_BOUND (default=1e6)
	- The minimum size of a "big array."
	- When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads are used for reduction.
* MXNET_ENABLE_GPU_P2P (default=1)
    - If true, MXNet tries to use GPU peer-to-peer communication, if available,
      when kvstore's type is `device`

## Memonger

* MXNET_BACKWARD_DO_MIRROR (default=0)
    - whether do `mirror` during training for saving device memory.
    - when set to `1`, then during forward propagation, graph executor will `mirror` some layer's feature map and drop others, but it will re-compute this dropped feature maps when needed. `MXNET_BACKWARD_DO_MIRROR=1` will save 30%~50% of device memory, but retains about 95% of running speed.
    - one extension of `mirror` in MXNet is called [memonger technology](https://arxiv.org/abs/1604.06174), it will only use O(sqrt(N)) memory at 75% running speed.

## Control the profiler

When USE_PROFILER is enabled in Makefile or CMake, the following environments can be used to profile the application without changing code. Execution options may affect the granularity of profiling result. If you need profiling result of every operator, please set MXNET_EXEC_BULK_EXEC_INFERENCE and MXNET_EXEC_BULK_EXEC_TRAIN to 0.

* MXNET_PROFILER_AUTOSTART (default=0)
	- Set to 1, MXNet starts the profiler automatically. The profiling result is stored into profile.json in the working directory.

* MXNET_PROFILER_MODE (default=0)
	- If set to '0', profiler records the events of the symbolic operators.
	- If set to '1', profiler records the events of all operators.

## Other Environment Variables

* MXNET_CUDNN_AUTOTUNE_DEFAULT (default=0)
    - The default value of cudnn_tune for convolution layers.
    - Auto tuning is turn off by default. For benchmarking, set this to 1 to turn it on by default.

Settings for Minimum Memory Usage
---------------------------------
- Make sure ```min(MXNET_EXEC_NUM_TEMP, MXNET_GPU_WORKER_NTHREADS) = 1```
  - The default setting satisfies this.

Settings for More GPU Parallelism
---------------------------------
- Set ```MXNET_GPU_WORKER_NTHREADS``` to a larger number (e.g., 2)
  - To reduce memory usage, consider setting ```MXNET_EXEC_NUM_TEMP```.
- This might not speed things up, especially for image applications, because GPU is usually fully utilized even with serialized jobs.
