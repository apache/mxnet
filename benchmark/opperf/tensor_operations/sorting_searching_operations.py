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

""" Performance benchmark tests for MXNet NDArray Sorting and Searching Operations

TODO

1. sort
2. argsort
3. topk
4. argmax
5. argmin
6. Sort and Argsort
    6.1 Descending Order
    6.2 Flatten and sort
7. TopK
    7.1 K being a very small number (ex: 1) on a axis with 1000 values.
8. argmax_channel (This is same as argmax with axis=-1)
"""