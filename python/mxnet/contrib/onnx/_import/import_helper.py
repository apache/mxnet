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
# pylint: disable=invalid-name
"""Operator attributes conversion"""
from .op_translations import _add, negative, reduce_max, reshape
from .op_translations import reduce_mean

# _convert_map defines maps of name to converter functor(callable)
_convert_map = {
    'Add'           : _add,
    'Neg'           : negative,
    'ReduceMax'     : reduce_max,
    'ReduceMean'    : reduce_mean,
    'Reshape'       : reshape,
}
