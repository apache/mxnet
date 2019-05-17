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

""" Performance benchmark tests for MXNet NDArray GEMM Operations

TODO

1. dot
2. batch_dot

3. As part of default tests, following needs to be added:
    3.1 Sparse dot. (csr, default) -> row_sparse
    3.2 Sparse dot. (csr, row_sparse) -> default
    3.3 With Transpose of lhs
    3.4 With Transpose of rhs
4. 1D array: inner product of vectors
"""
