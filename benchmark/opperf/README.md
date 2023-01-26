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

# MXNet Operator Performance Benchmarks

A Python utility for benchmarking and profiling individual MXNet operator execution.

With this utility, for each MXNet operator you can get the following details:

**Timing**
1. Forward execution time
2. Backward execution time

**Memory**
1. Average and Max memory allocated

NOTE: This is the `pool memory`. It does not reflect the exact memory requested by the operator.

# Motivation

Benchmarks are usually done end-to-end for a given Network Architecture. For example: ResNet-50 benchmarks on ImageNet data. This is good measurement of overall performance and health of a deep learning framework. However, it is important to note the following important factors:
1. Users use a lot more operators that are not part of a standard network like ResNet. Example: Tensor manipulation operators like mean, max, topk, argmax, sort etc.   
2. A standard Network Architecture like ResNet-50 is made up of many operators Ex: Convolution2D, Softmax, Dense and more. Consider the following scenarios:
    1. We improved the performance of Convolution2D operator, but due to a bug, Softmax performance went down. Overall, we may observe end to end benchmarks are running fine, we may miss out the performance degradation of a single operator which can accumulate and become untraceable.
    2. You need to see in a given network, which operator is taking maximum time and plan optimization work. With end to end benchmarks, it is hard to get more fine grained numbers at operator level.
3. We need to know on different hardware infrastructure (Ex: CPU with oneDNN, GPU with NVIDIA CUDA and cuDNN) how different operators performs. With these details, we can plan the optimization work at operator level, which could exponentially boost up end to end performance.
4. You want to have nightly performance tests across all operators in a deep learning framework to catch regressions early. 
5. We can integrate this framework with a CI/CD system to run per operator performance tests for PRs. Example: When a PR modifies the kernel of TransposeConv2D, we can run benchmarks of TransposeConv2D operator to verify performance.

Hence, in this utility, we will build the functionality to allow users and developers of deep learning frameworks to easily run benchmarks for individual operators.

# How to use

## Prerequisites

Provided you have MXNet installed (any version >= 1.5.1), all you need to use opperf utility is to add path to your cloned MXNet repository to the PYTHONPATH.

Note: 
1. Currently, opperf utility requires a cloned mxnet repo. It isn't supported on PyPi binary yet. [Work in Progress]
2. To install MXNet, refer [Installing MXNet page](https://mxnet.apache.org/versions/master/install/index.html)

```
export PYTHONPATH=$PYTHONPATH:/path/to/mxnet/
```

## Usecase 1 - Run benchmarks for all the operators

Below command runs all the MXNet operators (NDArray) benchmarks with default inputs and saves the final result as JSON in the given file.

```
python mxnet/benchmark/opperf/opperf.py --output-format json --output-file mxnet_operator_benchmark_results.json
```

**Other Supported Options:**

1. **output-format** : `json` or `md` for markdown file output.

2. **ctx** : `cpu` or `gpu`. By default, cpu on CPU machine, gpu(0) on GPU machine. You can override and set the global context for all operator benchmarks. Example: --ctx gpu(2).

3. **dtype** : By default, `float32`. You can override and set the global dtype for all operator benchmarks. Example: --dtype float64.

4. **profiler** : `native` or `python`. By default, 'native'. You can override and set the global profiler for all operator benchmarks. Example: --profiler 'python'.
Native profiler uses MXNet C++ based built-in profiler. Python profiler uses Python package time. Generally, native profiler is used by developers and python profiler is used by users.

5. **int64-tensor** : `on` or `off`. By default, 'off'. You can override and set the large tensor flag to ON. Example: --int64-tensor ON

## Usecase 2 - Run benchmarks for all the operators in a specific category

For example, you want to run benchmarks for all NDArray Broadcast Binary Operators, Ex: broadcast_add, broadcast_mod, broadcast_pow etc., You just run the following python script.

```
#!/usr/bin/python
from benchmark.opperf.nd_operations.binary_operators import run_mx_binary_broadcast_operators_benchmarks

# Run all Binary Broadcast operations benchmarks with default input values
print(run_mx_binary_broadcast_operators_benchmarks())
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
{'broadcast_mod': [{'avg_time_forward_broadcast_mod': 28.7063, 'avg_time_mem_alloc_cpu/0': 4194.3042,
                    'avg_time_backward_broadcast_mod': 12.0954, 'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}},
                   {'avg_time_forward_broadcast_mod': 2.7332, 'avg_time_mem_alloc_cpu/0': 400.0,
                    'avg_time_backward_broadcast_mod': 1.1288, 'inputs': {'lhs': (10000, 10), 'rhs': (10000, 10)}},
                   {'avg_time_forward_broadcast_mod': 30.5322, 'avg_time_mem_alloc_cpu/0': 4000.0,
                    'avg_time_backward_broadcast_mod': 225.0255, 'inputs': {'lhs': (10000, 1), 'rhs': (10000, 100)}}],
 'broadcast_power': [{'avg_time_backward_broadcast_power': 49.5871, 'avg_time_forward_broadcast_power': 18.0954,
                      'avg_time_mem_alloc_cpu/0': 4194.3042, 'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}},
                     {'avg_time_backward_broadcast_power': 4.6623, 'avg_time_forward_broadcast_power': 1.8283,
                      'avg_time_mem_alloc_cpu/0': 400.0, 'inputs': {'lhs': (10000, 10), 'rhs': (10000, 10)}},
                     {'avg_time_backward_broadcast_power': 279.922, 'avg_time_forward_broadcast_power': 24.4621,
                      'avg_time_mem_alloc_cpu/0': 4000.0, 'inputs': {'lhs': (10000, 1), 'rhs': (10000, 100)}}],
.....
.....                      
```

## Usecase 3 - Run benchmarks for specific operator
For example, you want to run benchmarks for `nd.add` operator in MXNet, you just run the following python script.

```
#!/usr/bin/python
import mxnet as mx
from mxnet import nd

from benchmark.opperf.utils.benchmark_utils import run_performance_test

add_res = run_performance_test(nd.add, run_backward=True, dtype='float32', ctx=mx.cpu(),
                               inputs=[{"lhs": (1024, 1024),
                                        "rhs": (1024, 1024)}],
                               warmup=10, runs=25)
print(add_res)
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
{'add': [{'avg_time_mem_alloc_cpu/0': 102760.4453,
          'avg_time_forward_broadcast_add': 4.0372,
          'avg_time_backward_broadcast_add': 5.3841,
          'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}}]}

```

## Usecase 4 - Run benchmarks for group of operators with same input
For example, you want to run benchmarks for `nd.add`, `nd.sub` operator in MXNet, with the same set of inputs. You just run the following python script.

```
#!/usr/bin/python
import mxnet as mx
from mxnet import nd

from benchmark.opperf.utils.benchmark_utils import run_performance_test

add_res = run_performance_test([nd.add, nd.subtract], run_backward=True, dtype='float32', ctx=mx.cpu(),
                               inputs=[{"lhs": (1024, 1024),
                                        "rhs": (1024, 1024)}],
                               warmup=10, runs=25)
print(add_res)
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
{'add': [{'avg_time_mem_alloc_cpu/0': 102760.4453,
          'avg_time_forward_broadcast_add': 4.0372,
          'avg_time_backward_broadcast_add': 5.3841,
          'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}}],
'subtract': [{'avg_time_forward_broadcast_sub': 5.5137, 
               'avg_time_mem_alloc_cpu/0': 207618.0469,
               'avg_time_backward_broadcast_sub': 7.2976, 
               'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}}
             ]}

```

## Usecase 5 - Profile internal operators locally
Currently, opperf supports operators in `mx.nd.*` namespace.
However, locally, one can profile internal operators in `mx.nd.internal.*` namespace.

## Usecase 6 - Compare performance for chosen operator from both NDArray library and its Numpy/Numpy_extension counterpart
For example, you want to compare add operator from `mx.nd` and `mx.np`. You just run the following python script.

```
#!/usr/bin/python
from benchmark.opperf.utils.benchmark_utils import run_benchmark_operator

run_benchmark_operator(name = "add", run_backward=True)
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
<module 'mxnet.ndarray'>
[{'add': [{'inputs': {'lhs': (128, 128), 'rhs': (128, 128)},
           'max_storage_mem_alloc_cpu/0': 32.768,
           'avg_time_forward_add': 0.0496,
           'avg_time_backward_add': 0.0793}]}]
<module 'mxnet.numpy'>
[{'add': [{'inputs': {'x1': (128, 128), 'x2': (128, 128)},
           'max_storage_mem_alloc_cpu/0': 32.768,
           'avg_time_forward_add': 0.0484,
           'avg_time_backward_add': 0.0898}]}]

```
This function uses `run_performance_test` function mentioned in Usecase 3 and Usecase 4 and it is possible to change all parameters from it.
All arguments that are of type NDArray will be automatically provided with shape that is passed as `size`.
If any fuction requires more arguments or different shaped NDArrays, provide those arguments as `additional_inputs` as it is shown below:
```
run_benchmark_operator(name = "pick", size = (128,128), additional_inputs = {"index": (128,1)})
```


#### Changes
Remove the hasattr check for `op.__name__` to be in `mx.nd`

The resulting diff would look like :
##### Old Code
```
-        if hasattr(mx.nd, op.__name__):
-            benchmark_result = _run_nd_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list, profiler)
-        else:
-            raise ValueError("Unknown NDArray operator provided to benchmark. -  ", op.__name__)
```
##### New Code
```
+        #if hasattr(mx.nd, op.__name__):
+        benchmark_result = _run_nd_operator_performance_test(op, inputs, run_backward, warmup, runs, kwargs_list, profiler)
+        #else:
+            #raise ValueError("Unknown NDArray operator provided to benchmark. -  ", op.__name__)
```

#### Result
This should allow profiling of any operator in MXNet provided user provides valid parameters [`inputs`, `run_backward`, etc] to the `run_performance_test` function.

#### Example
Provided the source code change is made in the `benchmark/opperf/utils/benchmark_utils.py`
```
>>> import mxnet as mx
>>> from mxnet import nd
>>> from benchmark.opperf.utils.benchmark_utils import run_performance_test
>>> run_performance_test(mx.nd._internal._copyto,inputs=[{"data":mx.nd.array([1,2]),"out":mx.nd.empty(shape=mx.nd.array([1,2]).shape,ctx=mx.cpu())}])
INFO:root:Begin Benchmark - _copyto
INFO:root:Complete Benchmark - _copyto
[{'_copyto': [{'inputs': {'data': '<NDArray 2 @cpu(0)>', 'out': '<NDArray 2 @cpu(0)>'}, 'max_storage_mem_alloc_cpu/0': 0.004}]}]
```

# How does it work under the hood?

Under the hood, executes NDArray operator using randomly generated data. Use MXNet profiler to get summary of the operator execution:
1. Memory
2. Computation time (forward, backward)

See the design proposal document for more details - https://cwiki.apache.org/confluence/display/MXNET/MXNet+Operator+Benchmarks 

**NOTE:**

This utility queries MXNet operator registry to fetch all operators registered with MXNet, generate inputs and run benchmarks.
However, fully automated tests are enabled only for simpler operators such as - broadcast operators, element_wise operators etc... For the purpose of readability and giving more control to the users, complex operators such as convolution (2D, 3D), Pooling, Recurrent are not fully automated but expressed as default rules.
See `utils/op_registry_utils.py` for more details.

## Use python timer
Optionally, you could use the python time package as the profiler engine to caliberate runtime in each operator.
To use python timer for all operators, use the argument --profiler 'python':
```
python mxnet/benchmark/opperf/opperf.py --profiler='python'
```

To use python timer for a specific operator, pass the argument profiler to the run_performance_test method:
```
add_res = run_performance_test([nd.add, nd.subtract], run_backward=True, dtype='float32', ctx=mx.cpu(),
                               inputs=[{"lhs": (1024, 1024),
                                        "rhs": (1024, 1024)}],
                               warmup=10, runs=25, profiler='python')
```
By default, MXNet profiler is used as the profiler engine.


# TODO

All contributions are welcome. Below is the list of desired features:

1. ~~Cover all MXNet operators~~.
2. Enhance MXNet profiler with additional APIs to programmatically fetch and process profiler data.
3. Integration with CI/CD system to run operator benchmarks for PR builds, nightly builds.
4. Dashboards and other modes of presentation of results for analyzing and planning tasks such as operator performance improvements.
5. Randomized Tensor Shape generation for profiling to identify bottlenecks in the operators.
