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

"""Operators that fallback to official NumPy implementation for np.linalg."""


import numpy as onp


__all__ = [
    'cond',
    'lstsq',
    'matrix_power',
    'matrix_rank',
    'multi_dot',
    'qr',
]

cond = onp.linalg.cond
lstsq = onp.linalg.lstsq
matrix_power = onp.linalg.matrix_power
matrix_rank = onp.linalg.matrix_rank
multi_dot = onp.linalg.multi_dot
qr = onp.linalg.qr
