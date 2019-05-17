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

# MXNet Operator Performance Benchmarks

A Python utility for benchmarking and profiling individual MXNet operator execution.

# How to use

## Prerequisites

This utility uses MXNet profiler under the hood to fetch compute and memory metrics. Hence, you need to build MXNet with `USE_PROFILER=1` flag.

Make sure to build the flavor of MXNet, for example - with/without MKL, with CUDA 9 or 10.1 etc., on which you would like to measure operator performance.

## Usecase 1 - Run benchmarks for all the operators

Below command runs all the MXNet operators (NDArray and Gluon) benchmarks with default inputs and saves the final result as JSON in the provided file.

```
python incubator-mxnet/benchmark/opperf/opperf.py --output-format json --output-file mxnet_operator_benchmark_results.json
```

**Other Options:**

1. **output-format** : json or md for markdown file output or csv.

2. **ctx** : By default, cpu on CPU machine, gpu(0) on GPU machine. You can override and set the global context for all operator benchmarks. Example: --ctx gpu(2).

3. **dtype** : By default, float32. You can override and set the global dtype for all operator benchmarks. Example: --dtype float64.

## Usecase 2 - Run benchmarks for all the operators in a specific category

For example, you want to run benchmarks for all NDArray Arithmetic Operators, you just run the following python script.

```
#! /usr/bin/python
from benchmark.opperf.tensor_operations.arithmetic_operations import run_arithmetic_operators_benchmarks

# Run all Arithmetic operations benchmarks with default input values
print(run_arithmetic_operators_benchmarks())
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
{'subtract': [{'avg_time_forward_broadcast_sub': 5.5137, 
               'avg_time_mem_alloc_cpu/0': 207618.0469,
               'avg_time_backward_broadcast_sub': 7.2976, 
               'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}}
             ],
 'add': [{'avg_time_mem_alloc_cpu/0': 207618.0469,
          'avg_time_forward_broadcast_add': 4.309,
          'avg_time_backward_broadcast_add': 5.6063,
          'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}},
        ],
 'multiply': [{'avg_time_backward_broadcast_mul': 19.1712,
               'avg_time_mem_alloc_cpu/0': 207618.0469,
               'avg_time_forward_broadcast_mul': 6.4855, 
               'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}},
             ]
}
```

## Usecase 3 - Run benchmarks for specific operator
For example, you want to run benchmarks for `nd.add` operator in MXNet, you just run the following python script.

```
#! /usr/bin/python
import mxnet as mx
from mxnet import nd

from benchmark.opperf.utils.benchmark_utils import run_performance_test

add_res = run_performance_test(nd.add, run_backward=True, dtype='float32', ctx=mx.cpu(),
                               inputs=[{"lhs": (1024, 1024),
                                        "rhs": (1024, 1024)}],
                               warmup=10, runs=25)
```

Output for the above benchmark run, on a CPU machine, would look something like below:

```
{'add': [{'avg_time_mem_alloc_cpu/0': 102760.4453,
          'avg_time_forward_broadcast_add': 4.0372,
          'avg_time_backward_broadcast_add': 5.3841,
          'inputs': {'lhs': (1024, 1024), 'rhs': (1024, 1024)}}]}

```
# How does it work under the hood?

Under the hood, executes NDArray operator or a Gluon block using randomly generated data. Use MXNet profiler to get summary of operator execution:
1. Memory
2. Computation time

See design proposal document for more details - https://cwiki.apache.org/confluence/display/MXNET/MXNet+Operator+Benchmarks 

# TODO

All contributions are welcome. Below is the list of desired features:

1. Cover all MXNet operators.
2. Enhance MXNet profiler with additional APIs to programmatically fetch and process profiler data.
3. Integration with CI/CD system to run operator benchmarks for PR builds, nightly builds.
4. Dashboards and other modes of presentation of results for analyzing and planning tasks such as operator performance improvements.
5. Integration with tools such as [Hypothesis](https://hypothesis.readthedocs.io/en/latest/) for randomized input generation for profiling to identify bottlenecks in operators.
