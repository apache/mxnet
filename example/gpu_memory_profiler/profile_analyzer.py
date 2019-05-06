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
"""profile_analyzer.py

   Analyze the memory profile in 'SETME.py'.
"""
import csv
import logging


def _internal_categorize(tag, memory_alloc,
                         kw_dict,
                         memory_profile,
                         warn_kw_404):
    """
    Categorize the {tag:memory_alloc} pair according to the keyword dictionary.
    """
    def _truncate(tag):
        return tag[:50] + '..' if len(tag) > 50 else tag

    kw_matched = None

    for key, val in kw_dict.items():
        for kw in val:
            if kw in tag:
                if kw_matched is not None:
                    logging.warning("Tag %s is considered match for "
                                    "multiple entries (%s, %s) "
                                    "in the keyword dictionary.", tag, kw_matched, kw)
                else:
                    kw_matched = kw
                if key in memory_profile:
                    memory_profile[key][0] += memory_alloc
                    if _truncate(tag) in memory_profile[key][1]:
                        memory_profile[key][1][_truncate(tag)] += memory_alloc
                    else:
                        memory_profile[key][1][_truncate(tag)] = memory_alloc
                else:
                    memory_profile[key] = [memory_alloc, {_truncate(tag) : memory_alloc}]

    if kw_matched is None:
        if warn_kw_404 and memory_alloc > 1e-2:
            logging.warning("Tag %s (Alloc Size %f) cannot match any entry "
                            "in the keyword dictionary.", tag, memory_alloc)
        if 'Others' in memory_profile:
            memory_profile['Others'][0] += memory_alloc
            if tag in memory_profile['Others'][1]:
                memory_profile['Others'][1][tag] += memory_alloc
            else:
                memory_profile['Others'][1][tag] = memory_alloc
        else:
            memory_profile['Others'] = [memory_alloc, {tag : memory_alloc}]


def analyze(memory_profile_name,
            layer_kw_dict,
            data_struct_kw_dict,
            expected_sum,
            warn_kw_404=False):
    """
    Analyze the memory profile.
    """
    layer_wise_memory_profile, data_struct_wise_memory_profile = {}, {}

    with open(memory_profile_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        trackable_memory_alloc = 0.0
        tag, memory_alloc = "unknown", 0.0

        for row in reader:
            tag, memory_alloc = row[0], float(row[1]) / 1024
            _internal_categorize(tag, memory_alloc,
                                 layer_kw_dict,
                                 layer_wise_memory_profile,
                                 warn_kw_404)
            _internal_categorize(tag, memory_alloc,
                                 data_struct_kw_dict,
                                 data_struct_wise_memory_profile,
                                 warn_kw_404)
            trackable_memory_alloc += memory_alloc

    # sort the memory profile from large to small, with 'Others' at the end
    layer_wise_memory_profile = \
        sorted(layer_wise_memory_profile.items(),
               key=lambda kv: kv[1][0] if kv[0] != 'Others' else 0.0, reverse=True)
    data_struct_wise_memory_profile = \
        sorted(data_struct_wise_memory_profile.items(),
               key=lambda kv: kv[1][0] if kv[0] != 'Others' else 0.0, reverse=True)

    # If 'expected_sum' is provided in 'SETME.py', append another entry
    #   'Untrackable' at the end to show the discrepancy between
    #   the memory consumptions that are trackable versus the ground truth.
    if expected_sum is not None:
        layer_wise_memory_profile.append(('Untrackable', expected_sum - trackable_memory_alloc))
        data_struct_wise_memory_profile.append(('Untrackable', expected_sum - \
                trackable_memory_alloc))

    return layer_wise_memory_profile, data_struct_wise_memory_profile
