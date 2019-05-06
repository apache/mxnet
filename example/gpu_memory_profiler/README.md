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

# GPU Memory Profiler


## Motivation

Machine learning training tasks running on the GPUs are very frequently limited 
  by the GPU memory capacity, frontend programmers therefore need a GPU memory 
  profiler to understand where the memory goes.
The problems with the existing GPU memory profilers are that they are too high-level 
  (similar to `nvidia-smi` that only provides a sum of all the consumptions) 
  and hence make it challenging for the frontend programmers to get the big picture.

In this example, we provide instructions on how to use the **MXNet GPU memory profiler**.
We start by making the observation that most MXNet programmers have
  a very good habit of assigning names to computation symbols:

```Python
# Symbolic Graph Implementation of an LSTM Cell
i2h = symbol.FullyConnected(data=inputs, weight=self._iW, bias=self._iB,
                            num_hidden=self._num_hidden*4,
                            name='%si2h'%name)
h2h = symbol.FullyConnected(data=states[0], weight=self._hW, bias=self._hB,
                            num_hidden=self._num_hidden*4,
                            name='%sh2h'%name)
```

Those names contain rich information regarding the model that is being described 
  and should be leveraged during the GPU memory profiling phase.
Luckily, MXNet already has the ability of propagating this information 
  all the way down from the Python APIs to its C++ core `nnvm::Graph`.
When the graph executor tries to initialize its data entries, we extract such information 
  out from the computation graph to tag the data entries (of type `mxnet::NDArray`),
  and when those data entries are materialized we are able to propagate those names
  to the storage allocators and record them inside the logging files,
  which could be further used for visualization and/or analyzing purposes.

![MXNet-GPU_Memory_Profiler-Design](https://github.com/dmlc/web-data/blob/master/mxnet/example/gpu_memory_profiler/MXNet-GPU_Memory_Profiler-Design.png?raw=true)


## Instructions

*The video below was demoed at **SysML 2019** ([Demo #24](https://www.sysml.cc/doc/2019/demo_24.pdf)). It shows the expected behavior of the GPU memory profiler. Please kindly note that we have made several changes since the demo so the current version looks slightly different.*

![MXNet-GPU_Memory_Profiler](https://github.com/dmlc/web-data/blob/master/mxnet/example/gpu_memory_profiler/MXNet-GPU_Memory_Profiler.gif?raw=true)

*In order for the GPU memory profiler to be enabled, you need to **compile from source**.*

- Download the MXNet codebase and install the prerequisite software libraries, as is instructed
    [here](https://mxnet.incubator.apache.org/versions/master/install/build_from_source.html).

- Modify the `MXNET_ENABLE_STORAGE_TAGGING` flag in `include/storage_tag.h` to **1**,
    which controls the storage tagging (disabled by default),
    **and** `MXNET_USE_GPU_MEMORY_PROFILER` flag in `src/profiler/gpu_memory_profiler.h` **1**,
    which controls dumping the GPU memory allocation inforamtion (also disabled by default).
  Please note that you must have **BOTH flags** set to **1** to use the GPU memory profiler.

- Build the MXNet core library and its Python binding (please refer to the same webpage as above).

- Run your application, as normal, but with `MXNET_USE_GPU_MEMORY_PROFILER`
    environment variable set to **1**.
  You should be able to see MXNet telling you that "*MXNet has the GPU memory profiler enabled.*".

```bash
env MXNET_USE_GPU_MEMORY_PROFILER=1 MXNET_ENGINE_TYPE=NaiveEngine python3 ... # your application goes here
# We will explain the 'NaiveEngine' environment in the following subsection
```

- You shall be able to see two files created in your current working directory,
    namely `mxnet_gpu_memory_profiler_output.csv` and 
           `mxnet_gpu_memory_profiler_output.log`.
  The former stores the mapping between the storage tag and the allocation size (in `MiB`).
  The latter tracks the call stacks that lead to an unknown storage tag (for debugging purpose).

- Run the `two_bar_plot.py` and `pprint_top_entries.py` for plotting and analyzing 
    information on the profiled information.
  Please note that you need to customize the keyword dictionary and the path to 
    the memory profile `mxnet_gpu_memory_profiler_output.csv` in `SETME.py` 
    **first** before running the analyzing scripts.


## !Attention

*As you use the GPU memory profiler, please kindly pay close attention to the following.*

- The information reported by the GPU memory profiler depends on the symbol names
    you assign at the Python frontend.
  Therefore, it is important that you give meaningful names as you program your models.

- It is **STRONGLY RECOMMENDED** that you use the **NAIVE** engine and storage pool for the GPU memory profiling
    (by setting `MXNET_ENGINE_TYPE=NaiveEngine` and `MXNET_GPU_MEM_POOL_TYPE=Naive`) 
    as those make the profiling logs much more readable.

- The GPU memory profiler works for Gluon `HybridBlock`s as well, as it also implicitly creates 
    symbolic computation graphs under the hood after `hybridize()` activation.
  However, it **DOES NOT WORK WELL** with the imperative programming paradigm 
    (i.e., `NDArray` function calls).
  The reason is because the design of the GPU memory profiler, as is described in previous text,
    relies heavily on the name assignments, but most imperative function calls are anonymous 
    (i.e., they are usually invoked without the `name` parameter), which makes it hard to track.

- The memory profile analyzer will analyze according to your memory profile according to
    your provided keyword dictionary.
  In addition to the keywords provided, the analyzer will also append two keywords (or one 
    if you do not provide the `expected_sum`), namely `Others` and `Untrackable`.
  `Others` includes the allocation entries that do not belong to any keyword categories.
  `Untrackable` will only appear if the `expected_sum` has been given and it shows
    the discrepancy between the total trackable memory allocations versus the 
    ground truth (that is usually reported by `nvidia-smi`).
  Note taht there is **NO OVERLAP** between `Others` and `Untrackable`.
  For most of the machine learning workloads we have examined so far, `Untrackable` typically
    ranges from `0.5 GB` to `0.7 GB` and it can be caused by CUDA library allocations
    (that happen underneath their APIs) or memory fragmentations.


## Questions or Concerns

If you have any questions or suggestions regarding the memory profiler, please open
  a new issue on Github pinging @ArmageddonKnight.


## Acknowledgement

The MXNet GPU memory profiler was first developed by *Abhishek Tiwari* (@olympian94)
  as part of the [*TBD* benchmarking tools](https://github.com/tbd-ai/tbd-tools).
*Bojian Zheng* (@ArmageddonKnight) refactored the codebase and adjusted 
  several design choices of the GPU memory profiler.
*Geoffrey Yu* (@geoffxy) made the demo video.
All works are done at the computer system group at the University of Toronto,
  under the supervision of Professor *Gennaey Pekhimenko* (@gpekhimenko).
