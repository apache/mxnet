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

""" Performance benchmark tests for MXNet NDArray NN Convolution Operators

1. Conv2D

TODO

2. Conv1D
3. Conv1DTranspose
4. Conv2DTranspose

Under the hood uses mx.nd.convolution.

NOTE: Number of warmup and benchmark runs for convolution may need to be reduced as the computation
is heavy and within first 25 runs results stabilizes without variation.
"""
