# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import functools

from .common_utils import merge_map_list
from mxnet import profiler

"""
TODO: Below we are using logic of parsing the MXNet profiler output string to
fetch the benchmark results. Note that this is a temporary solution till we add 
a new utility API into MXNet profiler to get_summary(), reset(). All the below
parsing logic should be removed once these read APIs are available in Profiler.

"""


def _get_memory_profile(memory_profile_results):
    memory_profile = {}
    for line in memory_profile_results:
        if line.startswith("Memory:"):
            device_id = line.split()[1]
            avg_time_memory_alloc = float(line.split()[-1])
            memory_profile["max_storage_mem_alloc_" + device_id] = avg_time_memory_alloc

    return memory_profile


def _get_operator_profile(operator_name, operator_profile_results):
    operator_profile = {}
    for line in operator_profile_results:
        if operator_name in line or operator_name[:3] + " " in line:
            operation = line.split()[0]
            operation_avg_time = float(line.split()[-1])
            if "_backward" in operation:
                operator_profile["avg_time" + operation] = operation_avg_time
            else:
                operator_profile["avg_time_forward_" + operation] = operation_avg_time

    return operator_profile


def parse_profiler_dump(operator_name, profiler_dump):
    """Parse the MXNet profiler dump output, fetch Memory profile results and
    Operator compute profiler results.

    Parameters
    ----------
    profiler_dump: string
        MXNet profiler output from mx.profiler.dumps() API.

    Returns
    -------
    map, Memory and Compute profiler results.

    """
    if not profiler_dump:
        raise AssertionError("Invalid MXNet profiler output provided to parse!")

    """
    MXNet profiler output from mx.profiler.dumps() API looks like below. This function parses
    this string profiler output to fetch Memory and Compute metrics.
    
    Profile Statistics.
    Note that counter items are counter values and not time units.
    Device Storage
    =================
    Name                          Total Count        Time (ms)    Min Time (ms)    Max Time (ms)    Avg Time (ms)
    ----                          -----------        ---------    -------------    -------------    -------------
    Memory: cpu/0                         100     2097152.0000     1681915.8750     2097152.0000      207618.0469

    MXNET_C_API
    =================
    Name                          Total Count        Time (ms)    Min Time (ms)    Max Time (ms)    Avg Time (ms)
    ----                          -----------        ---------    -------------    -------------    -------------
    MXNDArrayFree                          49           1.1220           0.0170           0.0360           0.0229
    MXAutogradBackwardEx                   50          11.5460           0.1980           0.3360           0.2309
    MXNet C API Calls                     399           1.9990           1.6010           1.9990           0.1990
    MXImperativeInvokeEx                   50           4.4810           0.0700           0.1330           0.0896
    MXNDArrayWaitAll                       50         769.0570          14.0200          24.5030          15.3811
    MXAutogradSetIsTraining               100           0.0190           0.0000           0.0010           0.0002
    MXAutogradSetIsRecording              100           0.0400           0.0000           0.0010           0.0004
    MXNet C API Concurrency               798           0.0000           0.0000           0.0010           0.0005

    operator
    =================
    Name                          Total Count        Time (ms)    Min Time (ms)    Max Time (ms)    Avg Time (ms)
    ----                          -----------        ---------    -------------    -------------    -------------
    DeleteVariable                        196           1.4490           0.0040           0.0250           0.0074
    _backward_broadcast_add               100         521.2320           4.8070           8.5970           5.2123
    SetValueOp                            100         645.8060           5.8820          10.0380           6.4581
    broadcast_add                         100         394.8910           3.5230           5.8790           3.9489
    """

    # String Patterns to look out for when parsing
    memory_profile_result_start = "Device Storage"  # Helps identify start of Memory profile
    c_api_profile_result_start = "MXNET_C_API"  # Helps identify end of Memory profile
    operator_profile_result_start = "operator"  # Helps identify start of Operator profile

    memory_profile_results = []
    operator_profile_results = []

    # Parse lines corresponding to Memory and Computation profiling
    read_memory_profile = False
    read_operator_profile = False
    for line in profiler_dump.splitlines():
        if line.startswith(memory_profile_result_start):
            read_memory_profile = True
        elif line.startswith(operator_profile_result_start):
            read_operator_profile = True
        elif line.startswith(c_api_profile_result_start):
            read_memory_profile = False

        if read_memory_profile:
            memory_profile_results.append(line)
        elif read_operator_profile:
            operator_profile_results.append(line)

    # Prepare results
    memory_profile = _get_memory_profile(memory_profile_results)
    operator_profile = _get_operator_profile(operator_name, operator_profile_results)

    return merge_map_list([memory_profile, operator_profile])


def profile(func):
    """Decorator for profiling MXNet operation.
    Uses MXNet profiler to collect metrics on memory usage and execution time
    of the operation.

    Parameters
    ----------
    func:
        Operation to be executed and timed.

    Returns
    -------
    res, profiler output. res being an return values from operator execution.
    profiler output is a dictionary with summary of operation execution.
    Example output : { "add": [{"avg_time_mem_alloc_cpu/0": 207618.0469,
                                "avg_time_forward_broadcast_add": 4.204,
                                "avg_time_backward_broadcast_add": 5.6288,
                                "inputs": {
                                            "lhs": [1024, 1024],
                                            "rhs": [1024,1024]
                                          }]
                     }
    """

    @functools.wraps(func)
    def profile_it(*args, **kwargs):
        # Profile the operation
        profiler.set_config(profile_all=True, aggregate_stats=True)
        profiler.set_state('run')
        res = func(*args, **kwargs)
        profiler.set_state('stop')

        # Prepare the results
        profiler_dump = profiler.dumps(reset=True)

        # args[0] is assumed to operator name, if not found check for block name.
        # NOTE: This parameter should be removed when we get away from parsing
        # profiler output and start using new profiler APIs - get_summary(), reset()
        if len(args) > 0:
            operator_name = args[0].__name__
        elif 'block' in kwargs:
            operator_name = kwargs['block']._op_name
        else:
            raise ValueError("Unable to identify operator name to extract profiler output!")

        # Get the MXNet profile output
        profiler_output = parse_profiler_dump(operator_name, profiler_dump)
        return res, profiler_output

    return profile_it
