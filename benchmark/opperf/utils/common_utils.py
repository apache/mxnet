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

from collections import ChainMap

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
    return dict(ChainMap(*map_list))


def save_to_file(inp_dict, out_filepath, out_format='json'):
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

    """
    if out_format == 'json':
        # Save as JSON
        with open(out_filepath, "w") as result_file:
            json.dump(inp_dict, result_file, indent=4, sort_keys=True)
    elif out_format == 'md':
        # Save as md
        with open(out_filepath, "w") as result_file:
            result_file.write(_prepare_markdown(inp_dict))
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


def _prepare_op_benchmark_result(op, op_bench_result):
    operator_name = op
    avg_forward_time = "---"
    avg_backward_time = "---"
    max_mem_usage = "---"
    inputs = "---"
    for key, value in op_bench_result.items():
        if "avg_time_forward" in key:
            avg_forward_time = value
        elif "avg_time_backward" in key:
            avg_backward_time = value
        elif "max_storage_mem_alloc_" in key:
            max_mem_usage = value
        elif "inputs" in key:
            inputs = value
    return "| {} | {} | {} | {} | {} |".format(operator_name, avg_forward_time, avg_backward_time,
                                               max_mem_usage, inputs)


def _prepare_markdown(results):
    results_markdown = [
        "| Operator | Avg Forward Time (ms) | Avg. Backward Time (ms) | Max Mem Usage (Storage) (Bytes)"
        " | Inputs |",
        "| :---: | :---: | :---: | :---:| :--- |"]

    for op, op_bench_results in sorted(results.items(), key=itemgetter(0)):
        for op_bench_result in op_bench_results:
            results_markdown.append(_prepare_op_benchmark_result(op, op_bench_result))

    return os.linesep.join(results_markdown)
