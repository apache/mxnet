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
import unittest

import mxnet as mx
import sys

from subprocess import Popen, PIPE


def test_nvtx_ranges_present_in_profile():

    if not mx.context.num_gpus():
        unittest.skip('Test only applicable to machines with GPUs')

    # Build a system independent wrapper to execute simple_forward with nvprof
    # This requires nvprof to be on your path (which should be the case for most GPU workstations with cuda installed).
    simple_forward_path = os.path.realpath(__file__)
    simple_forward_path = simple_forward_path.replace('test_nvtx', 'simple_forward')

    process = Popen(["nvprof", sys.executable, simple_forward_path], stdout=PIPE, stderr=PIPE)
    (output, profiler_output) = process.communicate()
    process.wait()
    profiler_output = profiler_output.decode('ascii')

    # Verify that some of the NVTX ranges we should have created are present
    # Verify that we have NVTX ranges for our simple operators.
    assert "Range \"FullyConnected\"" in profiler_output
    assert "Range \"_zeros\"" in profiler_output

    # Verify that we have some expected output from the engine.
    assert "Range \"WaitForVar\"" in profiler_output


if __name__ == '__main__':
    import nose
    nose.runmodule()
