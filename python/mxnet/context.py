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
"""Context management API of mxnet."""
from warnings import warn
from .device import Device, _current, cpu, gpu, cpu_pinned  # pylint: disable=unused-import
from .device import num_gpus, gpu_memory_info  # pylint: disable=unused-import


def Context(*args, **kwargs):
    """This class has been deprecated. Please refer to ``device.Device``."""
    warn('Directly use Context class to construct a device will be deprecated. '
         'Please use Device class instead. ', DeprecationWarning)
    return Device(*args, **kwargs)

def current_context():
    """This function has been deprecated. Please refer to ``device.current_device``."""
    warn('Directly use current_context to get current device will be deprecated. '
         'Please use current_device method instead. ', DeprecationWarning)
    return _current.get()
