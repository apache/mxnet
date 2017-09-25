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
# pylint: disable=fixme, invalid-name, too-many-arguments, too-many-locals, too-many-lines
# pylint: disable=too-many-branches, too-many-statements
"""Profiler setting methods."""
from __future__ import absolute_import

import ctypes
from .base import _LIB, check_call, c_str

def profiler_set_config(mode='symbolic', filename='profile.json'):
    """Set up the configure of profiler.

    Parameters
    ----------
    mode : string, optional
        Indicates whether to enable the profiler, can
        be 'symbolic', or 'all'. Defaults to `symbolic`.
    filename : string, optional
        The name of output trace file. Defaults to 'profile.json'.
    """
    mode2int = {'symbolic': 0, 'all': 1}
    check_call(_LIB.MXSetProfilerConfig(
        ctypes.c_int(mode2int[mode]),
        c_str(filename)))

def profiler_set_state(state='stop'):
    """Set up the profiler state to record operator.

    Parameters
    ----------
    state : string, optional
        Indicates whether to run the profiler, can
        be 'stop' or 'run'. Default is `stop`.
    """
    state2int = {'stop': 0, 'run': 1}
    check_call(_LIB.MXSetProfilerState(ctypes.c_int(state2int[state])))

def dump_profile():
    """Dump profile and stop profiler. Use this to save profile
    in advance in case your program cannot exit normally."""
    check_call(_LIB.MXDumpProfile())
