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
""" SETME.py

    Please complete the TODO items of this file to for the GPU memory profiling.
"""

layer_kw_dict = \
    {
        # TODO Complete the layer-wise keyword dictionary
        #        with domain-specific knowledge.
        """E.g.,
        "RNN"       : ["rnn"],
        "Embed"     : ["embed"]
        """
    }

data_struct_kw_dict = \
    {
        # Most machine learning models share the same types of data structures.
        "Parameters"   : ["in_arg", "arg_grad", "optimizer_state"],
        "Data Entries" : ["data_entry"],
        "Workspace"    : ["workspace"]
    }

memory_profile_path = ""  # TODO Set the path to `mxnet_gpu_memory_profile.csv`.

expected_sum = None  # TODO [Optional] Set the `expected_sum` to what is reported by `nvidia-smi`.
                     # If this value is set, then the analyzer will automatically append
                     #   an 'untrackable' entry to the end of the profiling results.
