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

all: gemm_lib relu_lib transposecsr_lib transposerowsp_lib

gemm_lib:
	g++ -shared -fPIC -std=c++11 gemm_lib.cc -o libgemm_lib.so -I ../../../include/mxnet

relu_lib:
	nvcc -shared -std=c++11 -Xcompiler -fPIC relu_lib.cu -o librelu_lib.so -I ../../../include/mxnet

transposecsr_lib:
	g++ -shared -fPIC -std=c++11 transposecsr_lib.cc -o libtransposecsr_lib.so -I ../../../include/mxnet

transposerowsp_lib:
	g++ -shared -fPIC -std=c++11 transposerowsp_lib.cc -o libtransposerowsp_lib.so -I ../../../include/mxnet

clean:
	rm -rf libgemm_lib.so librelu_lib.so libtransposecsr_lib.so libtransposerowsp_lib.so
