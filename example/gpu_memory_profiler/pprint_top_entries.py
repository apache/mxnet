#!/usr/bin/python3
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
"""pprint_top_entries.py

   Pretty-print the top entries in the GPU memory profile.
   Run this file after setting the 'SETME.py'.
"""
import pprint
from SETME import layer_kw_dict, \
        data_struct_kw_dict, \
        memory_profile_path, expected_sum
from profile_analyzer import analyze


def pprint_top_entries(memory_profile, top_k=20):
    """
    Pretty-print the top entries.
    """
    for item in memory_profile:
        if item[0] == 'Untrackable':
            print("Untrackable Consumption: %f" % item[1])
            continue

        print("Keyword: %s, Total Consumption: %f" % (item[0], item[1][0]))
        print("Entries:")
        sorted_subentries = sorted(item[1][1].items(), \
                key=lambda kv: kv[1], reverse=True)
        pprint.pprint(sorted_subentries[:top_k if top_k < len(sorted_subentries) \
                else len(sorted_subentries)])

if __name__ == '__main__':
    layer_wise_memory_profile, data_struct_wise_memory_profile = \
        analyze(memory_profile_path,
                layer_kw_dict,
                data_struct_kw_dict,
                expected_sum)
    pprint_top_entries(layer_wise_memory_profile)
    pprint_top_entries(data_struct_wise_memory_profile)
