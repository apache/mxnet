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
"""conftest.py contains configuration for pytest."""

import gc
import platform

import mxnet as mx
import pytest


@pytest.fixture(autouse=True)
def check_leak_ndarray(request):
    garbage_expected = request.node.get_closest_marker('garbage_expected')
    if garbage_expected:  # Some tests leak references. They should be fixed.
        yield  # run test
        return

    if 'centos' in platform.platform():
        # Multiple tests are failing due to reference leaks on CentOS. It's not
        # yet known why there are more memory leaks in the Python 3.6.9 version
        # shipped on CentOS compared to the Python 3.6.9 version shipped in
        # Ubuntu.
        yield
        return

    del gc.garbage[:]
    # Collect garbage prior to running the next test
    gc.collect()
    # Enable gc debug mode to check if the test leaks any arrays
    gc_flags = gc.get_debug()
    gc.set_debug(gc.DEBUG_SAVEALL)

    # Run the test
    yield

    # Check for leaked NDArrays
    gc.collect()
    gc.set_debug(gc_flags)  # reset gc flags

    seen = set()
    def has_array(element):
        try:
            if element in seen:
                return False
            seen.add(element)
        except (TypeError, ValueError, NotImplementedError):  # unhashable
            pass

        if isinstance(element, mx.nd._internal.NDArrayBase):
            return element._alive  # We only care about catching NDArray's that haven't been freed in the backend yet
        elif isinstance(element, mx.sym._internal.SymbolBase):
            return False
        elif hasattr(element, '__dict__'):
            return any(has_array(x) for x in vars(element))
        elif isinstance(element, dict):
            return any(has_array(x) for x in element.items())
        else:
            try:
                return any(has_array(x) for x in element)
            except (TypeError, KeyError, RecursionError):
                return False

    assert not any(has_array(x) for x in gc.garbage), 'Found leaked NDArrays due to reference cycles'
    del gc.garbage[:]
