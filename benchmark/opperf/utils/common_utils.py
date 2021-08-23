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

import os
import json
from operator import itemgetter

import logging
logging.basicConfig(level=logging.INFO)


def merge_map_list(map_list):
    """Merge all the Map in map_list into one final Map.

    Useful when you have a list of benchmark result maps and you want to
    prepare one final map combining all results.

    Parameters
    ----------
    map_list: List[maps]
        List of maps to be merged.

    Returns
    -------
    map where all individual maps in the into map_list are merged

    """
    # Preserve order of underlying maps and keys when converting to a single map
    final_map = dict()

    for current_map in map_list:
        for key in current_map:
            final_map[key] =  current_map[key]

    return final_map


def save_to_file(inp_dict, out_filepath, out_format='json', runtime_features=None, profiler='native'):
    """Saves the given input dictionary to the given output file.

    By default, saves the input dictionary as JSON file. Other supported formats include:
    1. md

    Parameters
    ----------
    inp_dict: map
        Input dictionary to be saved
    out_filepath: str
        Output file path
    out_format: str, default 'json'
        Format of the output file. Supported options - 'json', 'md'. Default - json.
    runtime_features: map
        Dictionary of runtime_features.

    """
    if out_format == 'json':
        # Save as JSON
        with open(out_filepath, "w") as result_file:
            json.dump(inp_dict, result_file, indent=4, sort_keys=False)
    elif out_format == 'md':
        # Save as md
        with open(out_filepath, "w") as result_file:
            result_file.write(_prepare_markdown(inp_dict, runtime_features, profiler))
    else:
        raise ValueError("Invalid output file format provided - '{}'. Supported - json, md".format(format))


def get_json(inp_dict):
    """Converts a given dictionary to prettified JSON string.

    Parameters
    ----------
    inp_dict: map
        Input dictionary to be converted to JSON.

    Returns
    -------
    Prettified JSON string

    """
    return json.dumps(inp_dict, indent=4)


def _prepare_op_benchmark_result(op, op_bench_result, profiler):
    operator_name = op
    avg_forward_time = "---"
    avg_backward_time = "---"
    max_mem_usage = "---"
    inputs = "---"
    avg_time = "---"
    p50_time = "---"
    p90_time = "---"
    p99_time = "---"

    for key, value in op_bench_result.items():
        if "avg_time_forward" in key:
            avg_forward_time = value
        elif "avg_time_backward" in key:
            avg_backward_time = value
        elif "max_storage_mem_alloc_" in key:
            max_mem_usage = value
        elif "inputs" in key:
            inputs = value
        elif "avg_time" in key:
            avg_time = value
        elif "p50_time" in key:
            p50_time = value
        elif "p90_time" in key:
            p90_time = value
        elif "p99_time" in key:
            p99_time = value

    result = ""
    if profiler == "native":
        result = "| {} | {} | {} | {} | {} |".format(operator_name,
                 inputs, max_mem_usage, avg_forward_time, avg_backward_time)
    elif profiler == "python":
        result = "| {} | {} | {} | {} | {} | {} |".format(operator_name, avg_time, p50_time, p90_time, p99_time, inputs)
    return result


def _prepare_markdown(results, runtime_features=None, profiler='native'):
    results_markdown = []
    if runtime_features and 'runtime_features' in runtime_features:
        results_markdown.append("# Runtime Features")
        idx = 0
        for key, value in runtime_features['runtime_features'].items():
            results_markdown.append('{}. {} : {}'.format(idx, key, value))

    results_markdown.append("# Benchmark Results")
    if profiler == 'native':
        results_markdown.append(
            "| Operator | Inputs | Max Mem Usage (Storage) (Bytes) | Avg Forward Time (ms)"
            " | Avg. Backward Time (ms) |")
        results_markdown.append("| :---: | :---: | :---: | :---: | :---: |")
    elif profiler == 'python':
        results_markdown.append(
            "| Operator | Avg Time (ms) | P50 Time (ms) | P90 Time (ms) | P99 Time (ms) | Inputs |")
        results_markdown.append("| :---: | :---: | :---: | :---: | :---: | :---: |")

    for op, op_bench_results in sorted(results.items(), key=itemgetter(0)):
        for op_bench_result in op_bench_results:
            results_markdown.append(_prepare_op_benchmark_result(op, op_bench_result, profiler))

    return os.linesep.join(results_markdown)
