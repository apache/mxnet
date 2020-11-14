#!/usr/bin/env python3

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
# pylint: disable=arguments-differ

# This test checks if dynamic loading of library into MXNet is successful
# and checks the computation of an external operator

import mxnet as mx
import os

# check if operator exists
if hasattr(mx.nd, 'min_ex'):
    raise Exception('Operator already loaded')
else:
    print('Operator not registered yet')

# test loading library
if (os.name == 'posix'):
    path = os.path.abspath('build/libexternal_lib.so')
    mx.library.load(path, False)

# execute operator
print(mx.nd.min_ex())
print('Operator executed successfully')
