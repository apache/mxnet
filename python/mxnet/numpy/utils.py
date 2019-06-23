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

"""Util functions for the numpy module."""


from __future__ import absolute_import

import numpy as onp

__all__ = ['float16', 'float32', 'float64', 'uint8', 'int32', 'int8', 'int64', 'pi']

float16 = onp.float16
float32 = onp.float32
float64 = onp.float64
uint8 = onp.uint8
int32 = onp.int32
int8 = onp.int8
int64 = onp.int64

pi = onp.pi
