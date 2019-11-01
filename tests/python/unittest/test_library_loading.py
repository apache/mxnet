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
from mxnet.base import MXNetError
from mxnet.test_utils import download, is_cd_run

def check_platform():
    return platform.machine() not in ['x86_64', 'AMD64']

@unittest.skipIf(check_platform(), "not all machine types supported")
@unittest.skipIf(is_cd_run(), "continuous delivery run - ignoring test")
def test_library_loading():
    if (os.name=='posix'):
        lib = 'libsample_lib.so'
        if os.path.exists(lib):
            fname = lib
        elif os.path.exists('build/'+lib):
            fname = 'build/'+lib
        else:
            raise MXNetError("library %s not found " % lib)
    elif (os.name=='nt'):
        lib = 'libsample_lib.dll'
        if os.path.exists('windows_package\\lib\\'+lib):
            fname = 'windows_package\\lib\\'+lib
        else:
            raise MXNetError("library %s not found " % lib)

    fname = os.path.abspath(fname)
    mx.library.load(fname)
