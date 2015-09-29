Environment Variables
=====================
MXNet have several settings that can be changed via environment variable.
Usually you do not need to change these settings, but they are listed here for reference.

* MXNET_GPU_WORKER_NTHREADS (default=1)
  - Maximum number of threads that do the computation job on each GPU.
* MXNET_GPU_COPY_NTHREADS (default=1)
  - Maximum number of threads that do memory copy job on each GPU.
* MXNET_CPU_WORKER_NTHREADS (default=1)
  - Maximum number of threads that do the CPU computation job.
* MXNET_CPU_PRIORITY_NTHREADS (default=4)
	- Number of threads given to prioritized CPU jobs.
  * MXNET_EXEC_ENABLE_INPLACE (default=true)
  - Whether to enable inplace optimization in symbolic execution.
* MXNET_EXEC_MATCH_RANGE (default=10)
  - The rough matching scale in symbolic execution memory allocator.
  - Set this to 0 if we do not want to enable memory sharing between graph nodes(for debug purpose).
* MXNET_ENGINE_TYPE (default=ThreadedEnginePerDevice)
  - The type of underlying execution engine of MXNet.
  - List of choices
    - NaiveEngine: very simple engine that use master thread to do computation.
    - ThreadedEngine: a threaded engine that uses global thread pool to schedule jobs.
    - ThreadedEnginePerDevice: a threaded engine that allocates thread per GPU.
* MXNET_KVSTORE_REDUCTION_NTHREADS (default=4)
	- Number of threads used for summing of big arrays.
* MXNET_KVSTORE_BIGARRAY_BOUND (default=1e6)
	- The minimum size of "big array".
	- When the array size is bigger than this threshold, MXNET_KVSTORE_REDUCTION_NTHREADS threads will be used for reduction.
