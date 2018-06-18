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

# Copyright (c) 2015 by Contributors
# File: serialization.h
# Purpose: Functions to query GPU count, arch, etc.
# Author: Dick Carter

"""Provides information on the visible CUDA GPUs on the system."""
# pylint: disable=broad-except
# As a stand-alone program, it prints a list of unique cuda SM architectures
import ctypes as C
from ctypes.util import find_library

def cint(init_val=0):
    """create a C int with an optional initial value"""
    return C.c_int(init_val)

def int_addr(x):
    """given a c_int, return it's address as an int ptr"""
    x_addr = C.addressof(x)
    INTP = C.POINTER(C.c_int)
    x_int_addr = C.cast(x_addr, INTP)
    return x_int_addr

def checked_call(f, *args):
    """call a cuda function and check for success"""
    error_t = f(*args)
    assert error_t == 0, "Failing cuda call %s returns %s." % (f.__name__, error_t)

def find_cuda_lib(candidates):
    for candidate in candidates:
        try:
            lib = find_library(candidate)
            if lib is not None:
                return lib
        except Exception:
            pass
    return None

# Find cuda library in an os-independent way ('nvcuda' needed for Windows)
try:
    cuda = C.cdll.LoadLibrary(find_cuda_lib(['cuda', 'nvcuda']))
    checked_call(cuda.cuInit, cint(0))
except Exception:
    cuda = None

def get_device_count():
    """get number of cuda devices on the system"""
    if cuda is None:
        return 0
    else:
        device_count = cint()
        checked_call(cuda.cuDeviceGetCount, int_addr(device_count))
        return device_count.value

def get_sm_arch(device_id):
    """get SM architecture of the device at the given index"""
    major = cint()
    minor = cint()
    checked_call(cuda.cuDeviceComputeCapability, int_addr(major),
                 int_addr(minor),
                 cint(device_id))
    return 10 * major.value + minor.value

def unique_sm_arches():
    """returns a list of unique cuda SM architectures on the system"""
    archs = set()
    device_count = get_device_count()
    for device_id in range(device_count):
        archs.add(get_sm_arch(device_id))
    return sorted(archs)

# print a list of unique cuda SM architectures on the system
if __name__ == '__main__':
    print(' '.join(str(x) for x in unique_sm_arches()))
