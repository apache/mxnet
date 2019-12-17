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

all: subgraph_lib gemm_lib

gemm_lib:
	g++ -shared -fPIC -std=c++11 gemm_lib.cc -o libgemm_lib.so -I ../../../include/mxnet

subgraph_lib:
	g++ -shared -fPIC -std=c++11 subgraph_lib.cc -o libsubgraph_lib.so -I ../../../include/mxnet

clean:
	rm -rf libsubgraph_lib.so libgemm_lib.so
