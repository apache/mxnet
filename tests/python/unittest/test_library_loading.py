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

# This test checks if dynamic loading of library into MXNet is successful

import os
import platform
import unittest
import mxnet as mx
from mxnet.test_utils import download

def check_platform():
    return platform.machine() not in ['x86_64', 'AMD64']

@unittest.skipIf(check_platform(), "not all machine types supported")
def test_library_loading():
    if (os.name=='posix'):
        lib = 'mylib.so'
    elif (os.name=='nt'):
        lib = 'mylib.dll'

    fname = mx.test_utils.download('https://mxnet-demo-models.s3.amazonaws.com/lib_binary/'+lib)
    fname = os.path.abspath(fname)
    mx.library.load(fname)
