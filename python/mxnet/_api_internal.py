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
"""Namespace of internal API

The functions in this namespace are automatically exported from C++ side via PackedFunc
that is registered by "MXNET_REGISTER_*" macro. This way makes calling Python functions from C++
side very easily.

Each string starts with "_" in the "MXNET_REGISTER_*" macro is an internal API.

Acknowledgement: This file originates from incubator-tvm
"""
