#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

import mxnet as mx

# Need to update here
opset_to_onnx = {
    12 : '1.7',
    13 : '1.8',
}

def generate_op_support_matrix(opset):
    all_versions = range(12, opset+1, 1)
    from collections import defaultdict
    support = defaultdict(lambda: '')
    for ver in all_versions:
        ops = mx.onnx.get_operator_support(ver)
        for op in ops:
            support[op] += '%s ' % opset_to_onnx.get(ver, 'unknown version')
    md_string = '|MXNet Op|ONNX Version|\n|:-|:-:|\n'
    for i in support.items():
        md_string += '|%s|%s|\n' % i
    try:
        file_name = './op_support_matrix.md'
        with open(file_name, 'w') as f:
            f.write(md_string)
    except Exception as e:
        print('Error writing to file')

# Change the parameter to the highest ONNX version supported
generate_op_support_matrix(13)
