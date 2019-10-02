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

all:
	g++ -std=c++11 -shared -fPIC init_lib.cc -o libinit_lib.so -I ../../../include/mxnet

test:
	g++ -std=c++11 -O3 -o libtest libtest.cc -ldl -I ../../../include/mxnet

windows:
	cl /LD init_lib.cc

win_test:
	cl libtest.cc

clean:
	rm -rf *.so libtest
