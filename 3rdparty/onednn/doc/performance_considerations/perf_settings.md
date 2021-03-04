Configuring oneDNN for Benchmarking {#dev_guide_performance_settings}
=====================================================================

The following settings are recommended for measuring oneDNN performance using
benchdnn. When measuring performance using any deep learning framework, refer
to its benchmarking documentation. However, the approach outlined below is
true for almost any compute-intensive application.

# CPU

## Threading Runtimes

It is a common practice to affinitize each compute thread to its own CPU core
when benchmarking performance. The method to do this depends on the threading
library used.

TBB intentionally does not provide a mechanism to control affinity or number
of threads via environment variables. However, TBB does create threads based
on the number of CPUs in the process
[CPU affinity mask](https://en.wikipedia.org/wiki/Affinity_mask)
at the time the library is initialized. This means that some of the examples
below work for TBB as well. Additionally, TBB implements an observer mechanism
that can be used to
[affinitize threads](https://www.threadingbuildingblocks.org/docs/help/reference/task_scheduler/task_scheduler_observer.html).

This document focuses on OpenMP runtime that has portable controls for thread
affinity documented
[here](https://www.openmp.org/spec-html/5.0/openmpch6.html#x287-20510006).
It should be noted that the OpenMP runtime that comes with Microsoft Visual
studio does not support them nor does it provide any other ways to control
thread affinity.

## Benchmarking Settings

The general principles below are not operating system-specific. However, of
all operating systems supported by oneDNN only Linux has the
[numactl(8)](https://linux.die.net/man/8/numactl) utility that makes it easy
to demonstrate them. NUMA stands for non-uniform memory access which is the
typical architecture of the modern CPUs in which individual sockets have their
own memory with separate physical memory attached. NUMA configuration is
possible even within a single socket when a socket consists of multiple chips
or tiles or when sub-NUMA clustering configurations are enabled.

Also, many modern CPUs may have multiple hardware threads per CPU core
enabled. Such threads are usually exposed by OS as additional logical
processors (thus a system with 4 cores and 2 hardware threads per core has 8
logical processors). If this is the case, the recommendation is to use only
one of hardware threads per core.

There are three most important setup variants when benchmarking oneDNN on CPU:
- *Whole machine*. It is generally not recommended but can be used when it
  is not possible or desirable to split work among multiple instances.
- *Single NUMA domain*. This setup is the recommended one. It can be used for
  throughput-oriented inference and training with multiple instances.
- *Several cores within the same NUMA domain*. This setup is recommended for
  latency-oriented small- or single-minibatch cases.
This document does not discuss how to actually setup a multi-instance
environment.

### Whole Machine

Typically a modern server CPU is configured to have multiple NUMA domains.
When running benchmarks on a whole machine, it is best to instruct the OS to
interleave physical memory allocation between those domains. This way the
computations have a higher chance to access physical memory from a local
domain and thus there is less cross-node traffic. This also lowers run-to-run
variation.

~~~sh
$ export OMP_PROC_BIND=spread
$ export OMP_PLACES=threads
$ export OMP_NUM_THREADS=# number of cores in the system
$ numactl --interleave=all ./benchdnn ...
~~~

### Single NUMA Domain

Here we instruct `numactl` to affinitize process to NUMA domain 0 both in
terms of CPU and memory locality.

~~~sh
$ export OMP_PROC_BIND=spread
$ export OMP_PLACES=threads
$ export OMP_NUM_THREADS=# number of cores in NUMA domain 0
$ numactl --membind 0 --cpunodebind 0 ./benchdnn ...
~~~

### Several Cores Within a NUMA Domain

In this case we want to use `numactl` options from the single NUMA domain
scenario, but place OpenMP threads close one to another.

~~~sh
$ export OMP_PROC_BIND=close
$ export OMP_PLACES=threads
$ export OMP_NUM_THREADS=# desired number of cores to use
$ numactl --membind 0 --cpunodebind 0 ./benchdnn ...
~~~

Unfortunately, this does not work when there are multiple hardware threads per
CPU, OpenMP runtimes place multiple threads on each core with the settings
above. Moreover, there is no way to describe the desired configuration in
which there is only one OpenMP thread per core without listing the
corresponding logical processors explicitly on the command line via `numactl
--physcpubind=<list>` (not shown here) or using non-portable environment
variables supported by OpenMP runtimes based on the Intel OpenMP runtime
(Clang, Intel C/C++ Compiler):

~~~sh
$ export KMP_HW_SUBSET=1T # Use 1 hardware thread per core
$ export OMP_PROC_BIND=close
$ export OMP_PLACES=threads
$ export OMP_NUM_THREADS=# desired number of cores to use
$ numactl --membind 0 --cpunodebind 0 ./benchdnn ...
~~~

@note
    It is safe to set `KMP_HW_SUBSET=1T` even if the machine is configured
    with a single hardware thread per core. It also makes it unnecessary to
    set `OMP_NUM_THREADS` in all the scenarios but the last as the number of
    threads is then inferred from the total number of logical processors
    in the process CPU affinity mask.

