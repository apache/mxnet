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

import json

from collections import ChainMap


def merge_map_list(map_list):
    """Merge all the Map in map_list into one final Map.

    Useful when you have a list of benchmark result maps and you want to
    prepare one final map combining all results.

    :param map_list: List of maps to be merged.
    :return: map where all individual maps in the into map_list are merged

    """
    return dict(ChainMap(*map_list))


def save_to_file(inp_dict, out_filepath, out_format='json'):
    """Saves the given input dictionary to the given output file.

    By default, saves the input dictionary as JSON file. Other supported formats include:
    1. md
    2. csv

    :param inp_dict: Input dictionary to be saved
    :param out_filepath: Output file path
    :param out_format: Format of the output file. Supported options - 'json', 'md', 'csv'. Default - json.

    """
    if out_format == 'json':
        # Save as JSON
        with open(out_filepath, "w") as result_file:
            json.dump(inp_dict, result_file, indent=4)
    elif format == 'md' or format == 'csv':
        print("MD / CSV file output format not supported yet! Choose JSON")
    else:
        raise ValueError("Invalid output file format provided - '%s'. Supported - json, md, csv".format(format))


def get_json(inp_dict):
    """Converts a given dictionary to prettified JSON string.

    :param inp_dict: Input dictionary to be converted to JSON.
    :return: Prettified JSON string

    """
    return json.dumps(inp_dict, indent=4)
