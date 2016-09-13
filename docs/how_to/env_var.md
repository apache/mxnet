Environment Variables
=====================
MXNet have several settings that can be changed via environment variable.
Usually you do not need to change these settings, but they are listed here for reference.

## Set the number of threads

* MXNET_GPU_WORKER_NTHREADS (default=2)
  - Maximum number of threads that do the computation job on each GPU.
* MXNET_GPU_COPY_NTHREADS (default=1)
  - Maximum number of threads that do memory copy job on each GPU.
* MXNET_CPU_WORKER_NTHREADS (default=1)
  - Maximum number of threads that do the CPU computation job.
* MXNET_CPU_PRIORITY_NTHREADS (default=4)
	- Number of threads given to prioritized CPU jobs.

## Memory options

* MXNET_EXEC_ENABLE_INPLACE (default=true)
  - Whether to enable inplace optimization in symbolic execution.
* MXNET_EXEC_MATCH_RANGE (default=10)
  - The rough matching scale in symbolic execution memory allocator.
  - Set this to 0 if we do not want to enable memory sharing between graph nodes(for debug purpose).
* MXNET_EXEC_NUM_TEMP (default=1)
  - Maximum number of temp workspace we can allocate to each device.
  - Set this to small number can save GPU memory.
  - It will also likely to decrease level of parallelism, which is usually OK.

## Engine type

* MXNET_ENGINE_TYPE (default=ThreadedEnginePerDevice)
  - The type of underlying execution engine of MXNet.
  - List of choices
    - NaiveEngine: very simple engine that use master thread to do computation.
    - ThreadedEngine: a threaded engine that uses global thread pool to schedule jobs.
    - ThreadedEnginePerDevice: a threaded engine that allocates thread per GPU.

## Control the data communication

* MXNET_KVSTORE_REDUCTION_NTHREADS (default=4)
	- Number of CPU threads used for summing of big arrays.
* MXNET_KVSTORE_BIGARRAY_BOUND (default=1e6)
	- The minimum size of "big array".
	- When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads will be used for reduction.
* MXNET_ENABLE_GPU_P2P (default=1)
    - If true, mxnet will try to use GPU peer-to-peer communication if available
      when kvstore's type is `device`

## Others

* MXNET_CUDNN_AUTOTUNE_DEFAULT (default=0)
    - The default value of cudnn_tune for convolution layers.
    - Auto tuning is turn off by default. Set to 1 to turn on by default for benchmarking.

Settings for Minimum Memory Usage
---------------------------------
- Make sure ```min(MXNET_EXEC_NUM_TEMP, MXNET_GPU_WORKER_NTHREADS) = 1```
  - The default setting satisfies this.

Settings for More GPU Parallelism
---------------------------------
- Set ```MXNET_GPU_WORKER_NTHREADS``` to larger number (e.g. 2)
  - You may want to set ```MXNET_EXEC_NUM_TEMP``` to reduce memory usage.
- This may not speed things up, especially for image applications, because GPU is usually fully utilized even with serialized jobs.
