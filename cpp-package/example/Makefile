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

prebuild :
	@mkdir -p build
	$(shell ./get_data.sh)
	$(shell cp -r ../../lib ./)
CPPEX_SRC = $(wildcard *.cpp)
CPPEX_EXE = $(patsubst %.cpp, %, $(CPPEX_SRC))

CFLAGS += -I../../include -I../../3rdparty/tvm/nnvm/include -I../../3rdparty/dmlc-core/include  -I../include

ifeq ($(MXNET_USE_CPU),1)
	CFLAGS += -D MXNET_USE_CPU
endif

# CPPEX_CFLAGS += -I../include
CPPEX_EXTRA_LDFLAGS := -L../../lib -lmxnet

.PHONY: all clean

all: prebuild  $(CPPEX_EXE)

debug: CPPEX_CFLAGS += -DDEBUG -g
debug: prebuild all



$(CPPEX_EXE):% : %.cpp
	$(CXX) -std=c++11 $(CFLAGS)  $(CPPEX_CFLAGS) -o build/$@ $(filter %.cpp %.a, $^) $(CPPEX_EXTRA_LDFLAGS)

clean:
	@rm -rf build
