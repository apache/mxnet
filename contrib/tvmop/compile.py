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

# coding: utf-8
"""TVM Operator compile entry point"""
import tvm

import os
import argparse
from tvmop.opdef import __OP_DEF__

def get_target(device):
    if device == "cpu":
        return "llvm"
    elif device == "cuda" or device == "gpu":
        return "cuda"
    assert False, "Unknown device " + device


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    parser = argparse.ArgumentParser(description="Generate tvm operators")
    parser.add_argument("-o", action="store", required=True, dest="target_path",
                        help="Target path which stores compiled library")
    arguments = parser.parse_args()

    func_list_llvm = []
    func_list_cuda = []

    # TODO: attach instruction features to the library, e.g., avx-512, etc.
    for operator_def in __OP_DEF__:
        for sch, args, name in operator_def.invoke_all():
            if tvm.module.enabled(get_target(operator_def.target)):
                func_list = func_list_llvm if operator_def.target == "cpu" else func_list_cuda
                func_lower = tvm.lower(sch, args,
                                       name=name,
                                       binds=operator_def.get_binds(args))
                func_list.append(func_lower)

    lowered_funcs = {get_target("cpu") : func_list_llvm}
    if len(func_list_cuda) > 0:
        lowered_funcs[get_target("cuda")] = func_list_cuda
    func_binary = tvm.build(lowered_funcs, name="tvmop")
    func_binary.export_library(arguments.target_path)
