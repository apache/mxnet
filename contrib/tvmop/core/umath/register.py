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

from .core import *  # pylint: disable=wildcard-import
from . import operator as _op
from ...opdef import defop
from ...utils import AllTypes

unary_cpu_attrs = {
    'target': 'cpu',
    'dtype': AllTypes + ['bool'],
    'ndim': [1],
    'req': ['kWriteTo', 'kAddTo'],
    'attrs': ['req']
}

unary_gpu_attrs = {
    'target': 'gpu',
    'dtype': ["float32", "float64", "uint8", "int8", "int32", "int64", "bool"],
    'ndim': [1],
    'req': ['kWriteTo', 'kAddTo'],
    'attrs': ['req']
}


@defop(name="abs_cpu", **unary_cpu_attrs)
def abs_cpu(dtype, ndim, req):
    return unary_cpu(_op.abs, dtype, ndim, req)


@defop(name="abs_gpu", **unary_gpu_attrs)
def abs_gpu(dtype, ndim, req):
    return unary_gpu(_op.abs, dtype, ndim, req)
