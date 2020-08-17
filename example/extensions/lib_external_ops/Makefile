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

SRCS=init_lib.cc \
     min_ex.cc \
     ../../../src/lib_api.cc

INC_PATH=-I../../../include \
    -I../../../src/operator/

LIB_PATH=-L../../../build

LIBS=-lmxnet

all:
	g++ -shared -fPIC -std=c++11 init_lib.cc min_ex.cc.o ../../../src/lib_api.cc -o libmin_ex.so -I../../../include -L../../../build -lmxnet

test:
	g++ -shared -fPIC -std=c++11 ${SRCS} -o libmin_ex.so ${INC_PATH} ${LIB_PATH} ${LIBS}

clean:
	rm -rf libmin_ex.so
