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
"""Library management API of mxnet."""
from __future__ import absolute_import
import ctypes
import os
from .base import _LIB, check_call, MXNetError

def load(path):
    """Loads library dynamically.

    Parameters
    ---------
    path : Path to library .so/.dll file

    Returns
    ---------
    void
    """
    #check if path exists
    if not os.path.exists(path):
        raise MXNetError("load path %s does NOT exist" % path)
    #check if path is an absolute path
    if not os.path.isabs(path):
        raise MXNetError("load path %s is not an absolute path" % path)
    #check if path is to a library file
    _, file_ext = os.path.splitext(path)
    if not file_ext in ['.so', '.dll']:
        raise MXNetError("load path %s is NOT a library file" % path)

    byt_obj = path.encode('utf-8')
    chararr = ctypes.c_char_p(byt_obj)
    check_call(_LIB.MXLoadLib(chararr))
